"""Shared machinery for the terminal recorder actor.

A recorder is a sink-only MUSCLE3 actor that drains an arbitrary number of
independent *timelines* (one per connected ``S`` port) and writes each to disk
via a pluggable :class:`Sink`. :mod:`recorder_component` picks the sink by a
``format`` setting; everything the formats share lives here:

- **Dynamic ports** (MUSCLE3 0.10): the instance is created without a port
  description, so ports come from the yMMSL config. Any connected ``S`` port
  whose name maps to an IDS (optional ``_in`` suffix stripped) is drained.
- **One worker thread per sender**, so a slow timeline never head-of-line
  blocks a busy one. Ports from one sender share a thread (they share a
  libmuscle ``MPPClient``, which is not concurrency-safe). Receives go through
  the communicator, not ``instance.receive`` (which shuts the whole instance
  down on any one port's close), and the deadlock detector is disabled.

A timeline ends when its peer's port *closes*; ``next_timestamp is None`` is an
intermediate stream restart (an outer-loop iteration boundary), not the end.
"""

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from imas import IDSFactory
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, InstanceFlags, Message
from libmuscle.mpp_message import ClosePort
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import get_setting_optional

logger = logging.getLogger()


def ids_name_from_port(port_name: str) -> str:
    """Map a port name to the IDS name to deserialize it as.

    The port name is the IDS name, with an optional trailing ``_in`` stripped.
    Raises if the result is not a valid IDS name.
    """
    ids_name = port_name[:-3] if port_name.endswith("_in") else port_name
    if ids_name not in IDSFactory().ids_names():
        raise ValueError(
            f"Port '{port_name}' does not map to a known IDS name "
            f"(resolved to '{ids_name}'). Name the port after the IDS it "
            f"carries, optionally with an '_in' suffix."
        )
    return ids_name


def precompute_ids_metadata(ids_names: Iterable[str]) -> None:
    """Build each IDS type's metadata once, single-threaded.

    imas-python caches it lazily on first construction, which is not
    thread-safe; doing it here lets the worker threads only ever read it.
    """
    factory = IDSFactory()
    for ids_name in set(ids_names):
        factory.new(ids_name)


def ids_from_message(ids_name: str, data: bytes) -> IDSToplevel:
    """Deserialize a received message's payload into a fresh IDS."""
    ids = IDSFactory().new(ids_name)
    ids.deserialize(data)
    return ids


class TimelineHandler(Protocol):
    """Records one timeline's messages. One per port, used by one thread (so it
    need not be thread-safe); built by a factory given to serve_timelines."""

    def handle(self, seq: int, msg: Message) -> str:
        """Record message ``seq``; return a short detail to log (a URI)."""
        ...

    def close(self) -> None:
        """Release per-timeline resources after the timeline ends."""
        ...


#: A factory mapping ``(port, ids_name)`` to a handler for that timeline.
HandlerFactory = Callable[[str, str], TimelineHandler]


class Sink(Protocol):
    """Writes one occurrence's messages to disk in a given format.

    Built per occurrence by a :data:`SinkFactory`; each decodes a message only
    as far as its format needs (raw bytes, an IDS, distilled arrays). Used by
    one thread, so it need not be thread-safe.
    """

    def write(self, msg: Message) -> str:
        """Write one message; return a short detail to log."""
        ...

    def close(self) -> None:
        """Finalize this occurrence's store."""
        ...


#: Builds a :class:`Sink` for one occurrence, given its store base path (no
#: suffix) and the IDS name carried by the timeline.
SinkFactory = Callable[[Path, str], Sink]


class OccurrenceRecorder:
    """A :class:`TimelineHandler` that writes each outer-loop iteration to its
    own occurrence store ``<store_dir>/<NNNN>``, rolling on a stream restart (a
    message with no ``next_timestamp``, or time stepping backwards).
    All format-specific work lives in the injected :data:`SinkFactory`.
    """

    def __init__(
        self, store_dir: Path, ids_name: str, make_sink: SinkFactory
    ) -> None:
        self._store_dir = store_dir
        self._ids_name = ids_name
        self._make_sink = make_sink
        self._occurrence = 0
        self._sink: Optional[Sink] = None
        self._last_time: Optional[float] = None
        self._prev_ended = False

    def handle(self, seq: int, msg: Message) -> str:
        restarted = self._prev_ended or (
            self._last_time is not None and msg.timestamp < self._last_time
        )
        if self._sink is not None and restarted:
            self._sink.close()
            self._occurrence += 1
            self._sink = None
        if self._sink is None:
            base = self._store_dir / f"{self._occurrence:04d}"
            self._sink = self._make_sink(base, self._ids_name)
        detail = self._sink.write(msg)
        self._last_time = msg.timestamp
        self._prev_ended = msg.next_timestamp is None
        return detail

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()


def connected_s_ports(instance: Instance) -> List[str]:
    """Sorted, connected ``S`` ports of a terminal recorder, validated.

    Raises if the instance has any non-``S`` ports, since it is terminal.
    """
    ports = instance.list_ports()
    for operator in (Operator.O_I, Operator.O_F, Operator.F_INIT):
        if ports.get(operator):
            raise RuntimeError(
                f"A recorder is terminal and only supports S ports; "
                f"got {operator.name} ports {ports.get(operator)}."
            )
    return sorted(
        p for p in ports.get(Operator.S, []) if instance.is_connected(p)
    )


def serve_timelines(
    s_ports: List[str],
    ids_names: Dict[str, str],
    handler_factory: HandlerFactory,
    instance: Instance,
) -> Dict[str, BaseException]:
    """Drain every timeline concurrently, one worker thread per sender.

    Ports fed by *distinct* senders drain in parallel threads, each blocking on
    its own ``receive``; ports sharing a sender drain on one thread (see
    :func:`_group_ports_by_peer`). Each ``receive`` goes through the
    communicator and the ``ClosePort`` is detected here, so one timeline ending
    does not cut off the others (``instance.receive`` would shut the whole
    instance down). A ``next_timestamp is None`` arrives as a normal message,
    so a worker does not stop there.

    Returns a mapping of port -> exception for any failed timeline (empty on
    success); handlers are always closed.
    """
    # Build every IDS type's metadata once on the main thread (imas-python's
    # lazy cache is not thread-safe); workers then only read it.
    precompute_ids_metadata(ids_names.values())
    _disable_deadlock_detector(instance)

    handlers = {p: handler_factory(p, ids_names[p]) for p in s_ports}
    errors: Dict[str, BaseException] = {}
    errors_lock = threading.Lock()
    counts: Dict[str, int] = {p: 0 for p in s_ports}

    try:
        workers = [
            threading.Thread(
                target=_drain_ports,
                args=(ports, handlers, counts, instance, errors, errors_lock),
                name=f"recorder-{ports[0]}",
            )
            for ports in _group_ports_by_peer(instance, s_ports)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        for port, handler in handlers.items():
            try:
                handler.close()
            except BaseException as exc:  # noqa: B036
                errors.setdefault(port, exc)
                logger.exception(
                    "closing handler for timeline '%s' failed", port
                )

    logger.info(
        "recorder: %d message(s) across %d timeline(s)",
        sum(counts.values()),
        len(counts),
    )
    return errors


def _disable_deadlock_detector(instance: Instance) -> None:
    """Turn off libmuscle's receive-timeout deadlock detector for this actor.

    The detector tracks a single waiting receive per instance, so the
    concurrent waits of the per-sender drain threads would crash it. A terminal
    recorder only receives, so it can never be part of a deadlock cycle.
    Best-effort: if the hook is gone, set
    ``<recorder>.muscle_deadlock_receive_timeout: -1.0`` in the workflow.
    """
    try:
        instance._communicator.set_receive_timeout(-1.0)
    except Exception:
        logger.warning(
            "could not disable the deadlock detector programmatically; set "
            "<recorder>.muscle_deadlock_receive_timeout: -1.0 in the workflow."
        )


def _group_ports_by_peer(
    instance: Instance, s_ports: List[str]
) -> List[List[str]]:
    """Group ports by the sender instance that feeds them.

    Ports from one sender share a single ``MPPClient`` (whose ``receive`` is
    not concurrency-safe), so they drain on one thread; ports from different
    senders drain concurrently. Falls back to one group per port if peer info
    cannot be read.
    """
    try:
        from ymmsl.v0_2 import Identifier

        peer_info = instance._communicator._peer_info
        groups: Dict[str, List[str]] = {}
        for port in s_ports:
            endpoints = peer_info.get_peer_endpoints(Identifier(port), [])
            groups.setdefault(str(endpoints[0].instance()), []).append(port)
        return list(groups.values())
    except Exception:
        logger.warning(
            "could not read peer info; draining one thread per port. If two "
            "ports share a sender that is unsafe -- give each its recorder."
        )
        return [[p] for p in s_ports]


def _drain_ports(
    ports: List[str],
    handlers: Dict[str, TimelineHandler],
    counts: Dict[str, int],
    instance: Instance,
    errors: Dict[str, BaseException],
    errors_lock: "threading.Lock",
) -> None:
    """One worker thread: round-robin ``ports`` (one sender) to their closes.

    With a single port per sender (the usual case) this is just that port's
    blocking receive loop; co-located ports take turns so the shared connection
    is never used concurrently.
    """
    active = list(ports)
    seq = {p: 0 for p in ports}
    while active:
        for port in list(active):
            try:
                msg, _ = instance._communicator.receive_message(port)
            except (RuntimeError, OSError) as exc:
                # A genuine mid-stream failure (peer crashed without a
                # ClosePort, or a torn socket). The data so far is on disk and
                # the manager reports a real crash via exit codes, so we end
                # this timeline and let the other workers carry on.
                active.remove(port)
                logger.warning(
                    "timeline '%s' ended after %d messages: %r",
                    port,
                    seq[port],
                    exc,
                )
                continue
            if isinstance(msg.data, ClosePort):
                active.remove(port)
                logger.info(
                    "timeline '%s' closed after %d messages", port, seq[port]
                )
                continue
            try:
                detail = handlers[port].handle(seq[port], msg)
            except BaseException as exc:  # noqa: B036  -- surfaced to main
                with errors_lock:
                    errors[port] = exc
                logger.exception(
                    "timeline '%s' failed after %d messages", port, seq[port]
                )
                active.remove(port)
                continue
            counts[port] += 1
            logger.info("handled %s t=%.4e -> %s", port, msg.timestamp, detail)
            seq[port] += 1


@dataclass
class RecorderSettings:
    """Settings shared by every recorder format.

    Read once (constant across reuses); a format reads any extra settings
    itself in its :data:`FactoryBuilder`.
    """

    store_path: Path


def read_recorder_settings(instance: Instance) -> RecorderSettings:
    """Read the common recorder settings.

    ``store_path`` defaults to the instance's run folder.
    """
    store_path_setting = get_setting_optional(instance, "store_path")
    store_path = (
        Path(store_path_setting)
        if store_path_setting is not None
        else Path.cwd()
    )
    return RecorderSettings(store_path=store_path)


# Builds the per-timeline HandlerFactory once the ports and common settings are
# known: (instance, settings, s_ports) -> HandlerFactory. This is where a
# format reads its extra settings (e.g. distill's auto/config).
FactoryBuilder = Callable[
    [Instance, RecorderSettings, List[str]], HandlerFactory
]


def run_recorder(name: str, build_factory: FactoryBuilder) -> None:
    """Run the shared MUSCLE3 reuse loop for the terminal recorder.

    ``build_factory(instance, settings, s_ports)`` is called once, after the
    connected ``S`` ports are known, and returns the :data:`HandlerFactory`
    used to make one :class:`TimelineHandler` per timeline. Every connected
    ``S`` port is drained to its real close, so this loop runs once even
    when the peer keeps reusing; per-occurrence bookkeeping lives in the
    handler, not here.
    """
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    settings: Optional[RecorderSettings] = None
    factory: Optional[HandlerFactory] = None
    while instance.reuse_instance():
        s_ports = connected_s_ports(instance)
        if not s_ports:
            # A recorder with nothing wired to it is a no-op, not an error.
            logger.warning(
                "%s recorder has no connected S ports; nothing to record.",
                name,
            )
            break
        # Validate all port -> IDS mappings up front so a bad config fails.
        ids_names = {p: ids_name_from_port(p) for p in s_ports}

        if settings is None:
            settings = read_recorder_settings(instance)
            settings.store_path.mkdir(parents=True, exist_ok=True)
            # Clear this recorder's own per-port dirs up front (never
            # store_path itself, which may be the run folder), so a re-run into
            # an explicit store_path leaves no stale occurrences behind.
            for port in s_ports:
                shutil.rmtree(settings.store_path / port, ignore_errors=True)
            factory = build_factory(instance, settings, s_ports)
        # settings and factory are set together on the first reuse.
        assert factory is not None

        logger.info(
            "%s recording %d timeline(s) %s to %s",
            name,
            len(s_ports),
            s_ports,
            settings.store_path,
        )

        errors = serve_timelines(s_ports, ids_names, factory, instance)

        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"{name} timeline(s) failed: {msg}")
            raise RuntimeError(f"{name} timeline(s) failed: {msg}")


def recorder_main(name: str, build_factory: FactoryBuilder) -> None:
    """Recorder module entry point: set up logging, then run the loop."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    run_recorder(name, build_factory)

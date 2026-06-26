"""Shared machinery for terminal *tap*-style actors.

A tap is a sink-only MUSCLE3 actor that drains an arbitrary number of
independent *timelines* (one per connected ``S`` port) and does something with
each received message. Two concrete taps build on this — :mod:`tap_component`
(records one raw DBEntry per message) and :mod:`distill_component` (distills
each message into compact scalars / profiles appended to a Zarr store) — and
everything they share lives here:

- **Dynamic ports** (MUSCLE3 0.10): the :class:`~libmuscle.Instance` is created
  without a port description, so the ports come from the yMMSL configuration.
  Any connected ``S`` port whose name maps to a valid IDS name (an optional
  ``_in`` suffix is stripped) is drained; see :func:`ids_name_from_port`.
- **Concurrent draining, one thread per sender**: each timeline is drained by
  its own worker thread, so a slow timeline never head-of-line blocks a busy
  one (ports from the same sender share a thread — they share one libmuscle
  ``MPPClient``, which is not concurrency-safe). Receives go through the
  communicator, not ``instance.receive`` (which would shut the whole instance
  down on the first port's close), and the deadlock detector is disabled for
  the tap; see :func:`serve_timelines`. A timeline ends when its peer's port
  *closes*; ``next_timestamp is None`` marks an intermediate stream restart,
  not the end (see ``reuse_and_close.md``).
- **Backpressure**: at shutdown the tap logs one line on what it recorded, and
  warns *once* if it was write-bound (spent most of its time writing rather
  than waiting for data) — meaning it is the bottleneck and senders back up.
  No live monitoring thread; just the wall-clock split it already measures.

A concrete tap supplies a :class:`TimelineHandler` factory to
:func:`serve_timelines`; the per-timeline loop, timing and error plumbing are
provided here.
"""

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from imas import IDSFactory
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, InstanceFlags, Message
from libmuscle.mpp_message import ClosePort
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import get_setting_optional

logger = logging.getLogger()

# Warn once at shutdown if the recorder spent at least this fraction of its
# time writing rather than waiting for data: it is the bottleneck and senders
# are backing up on it. A single diagnostic line, no live monitoring thread.
_WRITE_BOUND_FRACTION = 0.8


def ids_name_from_port(port_name: str) -> str:
    """Map a port name to the IDS name to deserialize it as.

    The port name is taken to be the IDS name, with an optional trailing
    ``_in`` suffix stripped. Raises if the result is not a valid IDS name.
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


def connected_s_ports(instance: Instance) -> List[str]:
    """Sorted, connected ``S`` ports of a terminal tap, validated.

    Raises if the instance has any non-``S`` ports, since a tap is terminal.
    """
    ports = instance.list_ports()
    for operator in (Operator.O_I, Operator.O_F, Operator.F_INIT):
        if ports.get(operator):
            raise RuntimeError(
                f"A tap actor is terminal and only supports S ports; "
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

    Each connected ``S`` port is a timeline. Ports fed by *distinct* senders
    are drained in parallel threads, each blocking on its own ``receive``, so a
    slow timeline never head-of-line blocks a busy one. Ports that share a
    sender are drained by one thread (see :func:`_group_ports_by_peer`): they
    share a single libmuscle ``MPPClient`` whose ``receive`` is not
    concurrency-safe.

    Each ``receive`` goes through the communicator, not ``instance.receive`` —
    the latter calls ``Instance.__shutdown()`` on the first port's close,
    severing this instance's connections to *every* peer; we detect each port's
    ``ClosePort`` ourselves so one timeline ending doesn't cut off the others
    (this also bypasses the MMSF sequence validator, which a terminal tap has
    no submodel loop to satisfy). Concurrent receives additionally need
    libmuscle's deadlock detector off — it asserts a single waiting receive per
    instance, so concurrent waits would crash the manager (see
    :func:`_disable_deadlock_detector`). A ``next_timestamp is None`` is an
    intermediate close that arrives as a normal message, so a worker does not
    stop there (see ``reuse_and_close.md``).

    Returns a mapping of port -> exception for any failed timeline (empty on
    success); the caller decides how to surface it. Handlers are always closed.
    """
    # Build every IDS type's metadata once, on the main thread: imas-python
    # caches it lazily on first construction and that is not thread-safe, so
    # the per-sender workers must only read it (see precompute_ids_metadata).
    precompute_ids_metadata(ids_names.values())
    _disable_deadlock_detector(instance)

    handlers = {p: handler_factory(p, ids_names[p]) for p in s_ports}
    errors: Dict[str, BaseException] = {}
    errors_lock = threading.Lock()
    # Per port [messages, seconds waiting on receive, seconds writing]: each
    # port is written by its own worker thread only, summed after the join.
    stats: Dict[str, List[float]] = {p: [0.0, 0.0, 0.0] for p in s_ports}

    try:
        workers = [
            threading.Thread(
                target=_drain_ports,
                args=(ports, handlers, stats, instance, errors, errors_lock),
                name=f"tap-{ports[0]}",
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

    _log_summary(stats)
    return errors


def _log_summary(stats: Dict[str, List[float]]) -> None:
    """Log one line on what was recorded, and warn if the tap was write-bound.

    ``stats`` maps each port to ``[messages, t_wait, t_write]``. If the tap
    spent most of its time writing rather than waiting for data it is the
    bottleneck and senders back up on it -- the one diagnostic worth flagging.
    """
    total = int(sum(s[0] for s in stats.values()))
    wait = sum(s[1] for s in stats.values())
    write = sum(s[2] for s in stats.values())
    logger.info(
        "recorder: %d message(s) across %d timeline(s)", total, len(stats)
    )
    busy = wait + write
    if total and busy > 0 and write / busy >= _WRITE_BOUND_FRACTION:
        logger.warning(
            "recorder is write-bound (%.0f%% of its time writing, not waiting "
            "for data): it may not keep up and senders back up on it.",
            100 * write / busy,
        )


def _disable_deadlock_detector(instance: Instance) -> None:
    """Turn off libmuscle's receive-timeout deadlock detector for this tap.

    The detector tracks a single waiting receive per instance (it asserts as
    much in the manager), so the concurrent waits of the per-sender drain
    threads would crash it. A terminal tap only receives, so it can never be
    part of a deadlock cycle, so disabling it is safe. Best-effort: if a
    libmuscle version no longer exposes the hook, set
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

    Ports from the same sender share one libmuscle ``MPPClient`` (one TCP
    connection, whose ``receive`` is not concurrency-safe), so they must drain
    on one thread; ports from different senders drain concurrently. Falls back
    to one group per port (assuming distinct senders) if libmuscle's peer info
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
    stats: Dict[str, List[float]],
    instance: Instance,
    errors: Dict[str, BaseException],
    errors_lock: "threading.Lock",
) -> None:
    """One worker thread: round-robin ``ports`` (one sender) to their closes.

    With a single port per sender (the usual case) this is just that port's
    blocking receive loop; co-located ports take turns so the shared connection
    is never used concurrently. Receives go through the communicator and the
    ClosePort is detected here (see :func:`serve_timelines`).
    """
    active = list(ports)
    seq = {p: 0 for p in ports}
    while active:
        for port in list(active):
            try:
                t0 = perf_counter()
                msg, _ = instance._communicator.receive_message(port)
                t1 = perf_counter()
            except (RuntimeError, OSError) as exc:
                # A genuine mid-stream failure (peer crashed without a
                # ClosePort, or a torn socket). The data so far is on disk and
                # a real crash is reported by the manager via exit codes, so we
                # end this timeline and let the other workers carry on.
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
                t2 = perf_counter()
            except BaseException as exc:  # noqa: B036  -- surfaced to main
                with errors_lock:
                    errors[port] = exc
                logger.exception(
                    "timeline '%s' failed after %d messages", port, seq[port]
                )
                active.remove(port)
                continue
            st = stats[port]
            st[0] += 1
            st[1] += t1 - t0  # waiting on receive
            st[2] += t2 - t1  # writing (handler)
            logger.info("handled %s t=%.4e -> %s", port, msg.timestamp, detail)
            seq[port] += 1


@dataclass
class RecorderSettings:
    """Settings shared by every recorder component.

    Read once (constant across reuses); a component reads any format-specific
    settings itself, in its :data:`FactoryBuilder`.
    """

    store_path: Path


def read_recorder_settings(instance: Instance) -> RecorderSettings:
    """Read the settings common to all recorder components.

    ``store_path`` defaults to the instance's run folder (its working directory
    in the MUSCLE3 run).
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
# component reads any format-specific settings (e.g. distill's auto/config).
FactoryBuilder = Callable[
    [Instance, RecorderSettings, List[str]], HandlerFactory
]


def run_recorder(name: str, build_factory: FactoryBuilder) -> None:
    """Run the shared MUSCLE3 reuse loop for a terminal recorder component.

    ``name`` labels log / error messages (e.g. ``"tap"``, ``"distill"``).
    ``build_factory`` is called once, after the connected ``S`` ports are known
    and the common settings read, as ``build_factory(instance, settings,
    s_ports)``; it returns the :data:`HandlerFactory` used to make one
    :class:`TimelineHandler` per timeline.

    Every connected ``S`` port is drained to its real close (see
    :func:`serve_timelines` and ``reuse_and_close.md``), so this loop runs
    effectively once even when the peer keeps reusing; per-occurrence /
    per-message bookkeeping lives in the handler, not here.
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
        # Validate all port -> IDS mappings up front so a bad config fails
        # fast, before any handler is built.
        ids_names = {p: ids_name_from_port(p) for p in s_ports}

        if settings is None:
            settings = read_recorder_settings(instance)
            settings.store_path.mkdir(parents=True, exist_ok=True)
            # Clear this recorder's own per-port dirs up front (never
            # store_path itself, which may be the instance's run folder), so a
            # re-run into an explicit store_path can't leave stale occurrences
            # behind. On the default fresh run folder this is a harmless no-op.
            for port in s_ports:
                shutil.rmtree(settings.store_path / port, ignore_errors=True)
            factory = build_factory(instance, settings, s_ports)
        # settings and factory are set together on the first reuse, so both are
        # non-None here for every iteration.
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
    """Module entry point for a recorder: set up logging, then run the loop."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    run_recorder(name, build_factory)

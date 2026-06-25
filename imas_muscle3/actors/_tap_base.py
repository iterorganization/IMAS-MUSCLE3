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
- **Backpressure monitoring**: a monitor thread logs the drain's aggregate
  *saturation* across the worker threads — the fraction of time spent handling
  (``t_write``) versus blocked waiting (``t_wait``); near 1 recording is the
  bottleneck and senders stall on the tap. Per-port message counts and handler
  costs are reported as diagnostics (see :class:`DrainMetrics`).

A concrete tap supplies a :class:`TimelineHandler` factory to
:func:`serve_timelines`; the per-timeline loop, timing, monitoring and error
plumbing are provided here.
"""

import logging
import shutil
import threading
from dataclasses import dataclass, field
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

# Exponential-moving-average weight for the rolling receive/handle timings.
_EWMA_ALPHA = 0.2

# The backpressure monitor is a fixed internal diagnostic, not a knob: it stays
# quiet unless the drain is saturated and emits one summary line at shutdown.
_MONITOR_INTERVAL = 30.0  # seconds between (quiet-unless-saturated) checks
_SATURATION_WARN = 0.8  # warn once the drain spends >=80% of its time handling


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
    """Build the IDS metadata for each name once, single-threaded.

    imas-python lazily builds and caches ``IDSMetadata`` the first time an IDS
    of a given type is constructed, and that construction is **not**
    thread-safe (concurrent first-construction races with
    ``AttributeError: type object 'IDSMetadata' has no attribute
    '__setattr__'``). Constructing each type once here populates the shared
    cache so the worker threads only ever read it, which makes concurrent
    deserialization and handling safe.
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
    """What :func:`serve_timelines` does with each message of one timeline.

    Implementations carry whatever per-port state they need (a store path, a
    Zarr sink, a distiller). They are constructed by a *factory* given to
    :func:`serve_timelines` and used by exactly one worker thread, so they need
    not be thread-safe themselves.
    """

    def handle(self, seq: int, msg: Message) -> str:
        """Handle message ``seq`` of this timeline.

        Returns a short human-readable detail (e.g. the URI written) that the
        worker logs alongside the timestamp.
        """
        ...

    def close(self) -> None:
        """Release per-timeline resources, once the timeline has ended."""
        ...


#: A factory mapping ``(port, ids_name)`` to a handler for that timeline.
HandlerFactory = Callable[[str, str], TimelineHandler]


@dataclass
class PortMetrics:
    """Thread-safe per-timeline counters (one S port), for diagnostics.

    Under the round-robin drain these are *not* a backpressure signal: a
    port's own ``receive`` rarely blocks (its next message is already
    buffered by the time the loop returns to it), so a per-port wait would
    mostly reflect scheduling. We keep only the unambiguous per-port facts —
    message count, last timestamp, and the mean time this port's handler
    takes (``t_write``, to spot a slow IDS) — and measure backpressure
    globally in :class:`DrainMetrics`.
    """

    port: str
    messages: int = 0
    last_timestamp: float = 0.0
    t_write_ewma: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, timestamp: float, t_write: float) -> None:
        """Record one message's timestamp and its handler's duration."""
        with self._lock:
            self.messages += 1
            self.last_timestamp = timestamp
            if self.messages == 1:
                self.t_write_ewma = t_write
            else:
                a = _EWMA_ALPHA
                self.t_write_ewma = a * t_write + (1 - a) * self.t_write_ewma

    def snapshot(self) -> str:
        """One-line human-readable summary for logging."""
        with self._lock:
            return (
                f"{self.port}: msgs={self.messages} "
                f"t_last={self.last_timestamp:.4e} "
                f"write={self.t_write_ewma * 1e3:.1f}ms"
            )


@dataclass
class DrainMetrics:
    """Thread-safe global backpressure for the single round-robin drain thread.

    That one thread is the shared resource, so backpressure is its property,
    not any port's. Its *saturation* is the fraction of recent loop time spent
    handling messages (``t_write``) versus blocked waiting for any port
    (``t_wait``): near 1 the drain never idles and senders stall on the tap;
    near 0 it mostly waits and exerts none. The global ``t_wait`` still
    under-counts a bit (a blocked receive on one port says nothing about the
    others' buffers), but "does the drain ever idle?" is the honest signal
    round-robin per-port timings cannot give.
    """

    messages: int = 0
    t_wait_ewma: float = 0.0
    t_write_ewma: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, t_wait: float, t_write: float) -> None:
        """Record one drained message's wait (receive) and handle timings."""
        with self._lock:
            self.messages += 1
            if self.messages == 1:
                self.t_wait_ewma = t_wait
                self.t_write_ewma = t_write
            else:
                a = _EWMA_ALPHA
                self.t_wait_ewma = a * t_wait + (1 - a) * self.t_wait_ewma
                self.t_write_ewma = a * t_write + (1 - a) * self.t_write_ewma

    @property
    def saturation(self) -> float:
        """Fraction of recent drain-loop time spent handling, not waiting."""
        with self._lock:
            denom = self.t_wait_ewma + self.t_write_ewma
            return self.t_write_ewma / denom if denom > 0 else 0.0


class BackpressureMonitor(threading.Thread):
    """Background thread that periodically logs drain backpressure."""

    def __init__(
        self,
        port_metrics: Dict[str, PortMetrics],
        drain: DrainMetrics,
        interval: float,
        saturation_warn: float,
    ) -> None:
        super().__init__(name="tap-monitor", daemon=True)
        self._port_metrics = port_metrics
        self._drain = drain
        self._interval = interval
        self._saturation_warn = saturation_warn
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            self._log()

    def stop(self) -> None:
        """Stop the monitor and emit a final summary."""
        self._stop.set()
        self._log(final=True)

    def _log(self, final: bool = False) -> None:
        saturation = self._drain.saturation
        hot = saturation >= self._saturation_warn
        # Stay quiet on the periodic tick unless the drain is handling-bound;
        # the final summary is always emitted (one line at shutdown).
        if not final and not hot:
            return

        total = sum(m.messages for m in self._port_metrics.values())
        prefix = "tap final summary" if final else "tap backpressure"
        logger.info(
            "%s: %d timelines, %d messages, drain saturation %.0f%%",
            prefix,
            len(self._port_metrics),
            total,
            saturation * 100,
        )
        if hot:
            logger.warning(
                "drain is handling-bound (saturation %.0f%% >= %.0f%%): "
                "senders may be stalling on the tap.",
                saturation * 100,
                self._saturation_warn * 100,
            )
        # Per-port handler costs are diagnostics (which IDS is slow), shown on
        # the final summary and whenever the drain is hot.
        for metric in self._port_metrics.values():
            logger.info("  %s", metric.snapshot())


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

    metrics = {p: PortMetrics(p) for p in s_ports}
    drain = DrainMetrics()
    handlers = {p: handler_factory(p, ids_names[p]) for p in s_ports}
    errors: Dict[str, BaseException] = {}
    errors_lock = threading.Lock()
    monitor = BackpressureMonitor(
        metrics, drain, _MONITOR_INTERVAL, _SATURATION_WARN
    )
    monitor.start()

    try:
        workers = [
            threading.Thread(
                target=_drain_ports,
                args=(
                    ports,
                    handlers,
                    metrics,
                    drain,
                    instance,
                    errors,
                    errors_lock,
                ),
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

    monitor.stop()
    return errors


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
    metrics: Dict[str, PortMetrics],
    drain: DrainMetrics,
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
            metrics[port].update(msg.timestamp, t2 - t1)
            drain.update(t1 - t0, t2 - t1)
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

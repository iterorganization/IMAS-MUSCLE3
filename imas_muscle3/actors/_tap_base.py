"""Shared machinery for terminal *tap*-style actors.

A tap is a sink-only MUSCLE3 actor that drains an arbitrary number of
independent *timelines* (one per connected ``S`` port) concurrently and does
something with each received message. The :mod:`tap_component` records the raw
IDS per message; the :mod:`distill_component` extracts distilled scalars /
profiles and appends them to a Zarr store. Everything they share lives here:

- **Dynamic ports** (MUSCLE3 0.10): the :class:`~libmuscle.Instance` is created
  without a port description, so the ports come from the yMMSL configuration.
  Any connected ``S`` port whose name maps to a valid IDS name (an optional
  ``_in`` suffix is stripped) is drained; see :func:`ids_name_from_port`.
- **One thread per timeline**: each connected ``S`` port is drained by its own
  thread so an idle timeline cannot head-of-line block a busy one. A timeline
  ends when its peer sends ``next_timestamp is None``.
- **Backpressure monitoring**: a monitor thread periodically logs, per timeline,
  the time spent blocked in ``receive`` (``t_wait``) versus the time spent
  handling the message (``t_write``), and a *saturation ratio*
  ``t_write / (t_wait + t_write)``. A ratio near 1 means the handler is the
  bottleneck and senders will stall on the tap.

A concrete tap supplies a :class:`TimelineHandler` factory to
:func:`serve_timelines`; the per-timeline loop, timing, monitoring and error
plumbing are provided here.
"""

import logging
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Protocol

from imas import IDSFactory
from libmuscle import Instance, Message
from ymmsl.v0_2 import Operator

logger = logging.getLogger()

# Exponential-moving-average weight for the rolling receive/handle timings.
_EWMA_ALPHA = 0.2


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
        """Release any per-timeline resources (called once the timeline ends)."""
        ...


#: A factory mapping ``(port, ids_name)`` to a handler for that timeline.
HandlerFactory = Callable[[str, str], TimelineHandler]


@dataclass
class PortMetrics:
    """Thread-safe rolling metrics for one timeline (one S port)."""

    port: str
    messages: int = 0
    last_timestamp: float = 0.0
    t_wait_ewma: float = 0.0
    t_write_ewma: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, timestamp: float, t_wait: float, t_write: float) -> None:
        """Record one message's wait/handle timings."""
        with self._lock:
            self.messages += 1
            self.last_timestamp = timestamp
            if self.messages == 1:
                self.t_wait_ewma = t_wait
                self.t_write_ewma = t_write
            else:
                a = _EWMA_ALPHA
                self.t_wait_ewma = a * t_wait + (1 - a) * self.t_wait_ewma
                self.t_write_ewma = a * t_write + (1 - a) * self.t_write_ewma

    @property
    def saturation(self) -> float:
        """Fraction of recent time spent handling rather than waiting.

        Near 0: the tap waits on the sender (no backpressure caused).
        Near 1: handling is the bottleneck and senders stall on the tap.
        """
        with self._lock:
            denom = self.t_wait_ewma + self.t_write_ewma
            return self.t_write_ewma / denom if denom > 0 else 0.0

    def snapshot(self) -> str:
        """One-line human-readable summary for logging."""
        with self._lock:
            return (
                f"{self.port}: msgs={self.messages} "
                f"t_last={self.last_timestamp:.4e} "
                f"wait={self.t_wait_ewma * 1e3:.1f}ms "
                f"write={self.t_write_ewma * 1e3:.1f}ms"
            )


class BackpressureMonitor(threading.Thread):
    """Background thread that periodically logs backpressure metrics."""

    def __init__(
        self,
        metrics: Dict[str, PortMetrics],
        interval: float,
        saturation_warn: float,
    ) -> None:
        super().__init__(name="tap-monitor", daemon=True)
        self._metrics = metrics
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
        total = sum(m.messages for m in self._metrics.values())
        prefix = "tap final summary" if final else "tap backpressure"
        logger.info(
            "%s: %d timelines, %d messages handled",
            prefix,
            len(self._metrics),
            total,
        )
        for metric in self._metrics.values():
            saturation = metric.saturation
            logger.info(
                "  %s saturation=%.0f%%", metric.snapshot(), saturation * 100
            )
            if saturation >= self._saturation_warn:
                logger.warning(
                    "  timeline '%s' is handling-bound (saturation %.0f%% "
                    ">= %.0f%%): senders may be stalling on the tap.",
                    metric.port,
                    saturation * 100,
                    self._saturation_warn * 100,
                )


def _worker(
    instance: Instance,
    port: str,
    handler: TimelineHandler,
    metric: PortMetrics,
    errors: Dict[str, BaseException],
) -> None:
    """Drain one timeline: receive, hand off and time each message until the
    peer signals the end of the timeline (``next_timestamp is None``).

    Handling runs fully in parallel across timelines; this is safe because
    :func:`precompute_ids_metadata` has warmed the imas-python metadata cache
    before any worker starts. Any exception is captured in ``errors`` keyed by
    port so the main loop can surface it; a dead worker would otherwise be
    invisible to ``join()``. The handler is always closed, even on error.
    """
    seq = 0
    try:
        while True:
            t0 = perf_counter()
            msg = instance.receive(port)
            t1 = perf_counter()
            detail = handler.handle(seq, msg)
            t2 = perf_counter()
            metric.update(msg.timestamp, t1 - t0, t2 - t1)
            logger.info("handled %s t=%.4e -> %s", port, msg.timestamp, detail)
            seq += 1
            if msg.next_timestamp is None:
                break
    except BaseException as exc:  # noqa: B036  -- re-raised from main
        errors[port] = exc
        logger.exception("timeline '%s' failed after %d messages", port, seq)
    finally:
        try:
            handler.close()
        except BaseException as exc:  # noqa: B036
            errors.setdefault(port, exc)
            logger.exception("closing handler for timeline '%s' failed", port)
    if port not in errors:
        logger.info("timeline '%s' finished after %d messages", port, seq)


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
    monitor_interval: float,
    saturation_warn: float,
) -> Dict[str, BaseException]:
    """Drain every timeline concurrently with a handler per port.

    Warms the imas-python metadata cache, starts the backpressure monitor,
    runs one worker thread per port (each owning a freshly built handler), and
    joins them. Returns a mapping of port -> exception for any failed timeline
    (empty on success); the caller decides how to surface it.
    """
    precompute_ids_metadata(ids_names.values())

    metrics = {p: PortMetrics(p) for p in s_ports}
    errors: Dict[str, BaseException] = {}
    monitor = BackpressureMonitor(metrics, monitor_interval, saturation_warn)
    monitor.start()

    threads = [
        threading.Thread(
            target=_worker,
            args=(
                instance,
                p,
                handler_factory(p, ids_names[p]),
                metrics[p],
                errors,
            ),
            name=f"tap-{p}",
        )
        for p in s_ports
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    monitor.stop()
    return errors

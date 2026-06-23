"""Tap recorder actor for MUSCLE3.

A terminal (sink-only) actor that taps onto an arbitrary number of independent
*timelines* and records every message it receives to disk, one DBEntry per
message.

Key properties:

- **Dynamic ports** (MUSCLE3 0.10): the :class:`~libmuscle.Instance` is created
  without a port description, so the ports come from the yMMSL configuration.
  The tap accepts *any* connected ``S`` port whose name is a valid IDS name
  (an optional ``_in`` suffix is stripped, so both ``equilibrium`` and
  ``equilibrium_in`` work). Each message is deserialized as that IDS.
- **One thread per timeline**: each connected ``S`` port is drained by its own
  thread so that an idle timeline cannot head-of-line block a busy one. A
  timeline ends when its peer sends a message with ``next_timestamp is None``.
  Recording runs in parallel; :func:`precompute_ids_metadata` warms the
  imas-python metadata cache up front so concurrent deserialization is safe.
- **Per-message DBEntries**: each received message is written to its own
  ``imas:hdf5?path=<store_path>/<port>/<seq>`` DBEntry, queryable afterwards
  with IMAS-Python. ``store_path`` defaults to the instance's run folder.
- **Backpressure monitoring**: a monitor thread periodically logs, per
  timeline, the time spent blocked in ``receive`` (``t_wait``) versus the
  time spent
  recording (``t_write``), and a *saturation ratio* ``t_write / (t_wait +
  t_write)``. A ratio near 1 means recording is the bottleneck and the senders
  will stall on the tap.

Example yMMSL (yMMSL v0.2)::

    components:
      tap:
        implementation: tap_component
        ports:
          s: [equilibrium_in, core_profiles_in]
    settings:
      tap.store_path: /scratch/tap_store
    implementations:
      tap_component:
        executable: python
        args: -u -m imas_muscle3.actors.tap_component
"""

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

from imas import DBEntry, IDSFactory
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from libmuscle import Instance, InstanceFlags
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import get_setting_optional

logger = logging.getLogger()

# Exponential-moving-average weight for the rolling receive/record timings.
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
        """Record one message's wait/record timings."""
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
        """Fraction of recent time spent recording rather than waiting.

        Near 0: the tap waits on the sender (no backpressure caused).
        Near 1: recording is the bottleneck and senders stall on the tap.
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
            "%s: %d timelines, %d messages recorded",
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
                    "  timeline '%s' is recording-bound (saturation %.0f%% "
                    ">= %.0f%%): senders may be stalling on the tap.",
                    metric.port,
                    saturation * 100,
                    self._saturation_warn * 100,
                )


def record_message(
    store_path: Path,
    port: str,
    ids_name: str,
    data: bytes,
    seq: int,
) -> str:
    """Deserialize one message and write it to its own DBEntry.

    Uses ``put`` for full / time-independent IDSs and ``put_slice`` for single
    time slices, mirroring :func:`imas_muscle3.data_sink_source.handle_sink`.

    Returns the IMAS URI the message was written to, so it can be logged for
    easy reopening.
    """
    ids = IDSFactory().new(ids_name)
    ids.deserialize(data)
    uri = f"imas:hdf5?path={store_path / port / f'{seq:08d}'}"
    with DBEntry(uri, "w") as entry:
        if (
            len(ids.time) > 1
            or ids.ids_properties.homogeneous_time == IDS_TIME_MODE_INDEPENDENT
        ):
            entry.put(ids)
        else:
            entry.put_slice(ids)
    return uri


def precompute_ids_metadata(ids_names: Iterable[str]) -> None:
    """Build the IDS metadata for each name once, single-threaded.

    imas-python lazily builds and caches ``IDSMetadata`` the first time an IDS
    of a given type is constructed, and that construction is **not**
    thread-safe (concurrent first-construction races with
    ``AttributeError: type object 'IDSMetadata' has no attribute
    '__setattr__'``). Constructing each type once here populates the shared
    cache so the worker threads only ever read it, which makes concurrent
    deserialization and recording safe.
    """
    factory = IDSFactory()
    for ids_name in set(ids_names):
        factory.new(ids_name)


def worker(
    instance: Instance,
    port: str,
    ids_name: str,
    store_path: Path,
    metric: PortMetrics,
    errors: Dict[str, BaseException],
) -> None:
    """Drain one timeline: receive, record and time each message until the
    peer signals the end of the timeline (``next_timestamp is None``).

    Recording runs fully in parallel across timelines; this is safe because
    :func:`precompute_ids_metadata` has warmed the imas-python metadata cache
    before any worker starts (see that function). Any exception is captured in
    ``errors`` keyed by port so the main loop can surface it; a dead worker
    would otherwise be invisible to ``join()``.
    """
    seq = 0
    try:
        while True:
            t0 = perf_counter()
            msg = instance.receive(port)
            t1 = perf_counter()
            uri = record_message(store_path, port, ids_name, msg.data, seq)
            t2 = perf_counter()
            metric.update(msg.timestamp, t1 - t0, t2 - t1)
            logger.info("recorded %s t=%.4e -> %s", port, msg.timestamp, uri)
            seq += 1
            if msg.next_timestamp is None:
                break
    except BaseException as exc:  # noqa: B036  -- re-raised from main
        errors[port] = exc
        logger.exception("timeline '%s' failed after %d messages", port, seq)
        return
    logger.info("timeline '%s' finished after %d messages", port, seq)


def main() -> None:
    """MUSCLE3 execution loop for the tap recorder."""
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    ports = instance.list_ports()
    for operator in (Operator.O_I, Operator.O_F, Operator.F_INIT):
        if ports.get(operator):
            raise RuntimeError(
                f"The tap recorder is terminal and only supports S ports; "
                f"got {operator.name} ports {ports.get(operator)}."
            )

    while instance.reuse_instance():
        s_ports = sorted(
            p for p in ports.get(Operator.S, []) if instance.is_connected(p)
        )
        if not s_ports:
            # A tap with nothing wired to it is a no-op, not an error: this
            # lets a tap be declared in a workflow but left unconnected until
            # someone wants to record a timeline.
            logger.warning(
                "tap recorder has no connected S ports; nothing to record."
            )
            break
        # Validate all port -> IDS mappings up front so a bad config fails
        # fast, before any worker thread is started.
        ids_names = {p: ids_name_from_port(p) for p in s_ports}
        # Warm the imas-python metadata cache single-threaded so the worker
        # threads can record in parallel safely.
        precompute_ids_metadata(ids_names.values())

        # store_path defaults to the instance's run folder (its working
        # directory in the MUSCLE3 run).
        store_path_setting = get_setting_optional(instance, "store_path")
        store_path = (
            Path(store_path_setting)
            if store_path_setting is not None
            else Path.cwd()
        )
        clean_on_start = get_setting_optional(instance, "clean_on_start", True)
        monitor_interval = get_setting_optional(
            instance, "monitor_interval", 5.0
        )
        saturation_warn = get_setting_optional(
            instance, "saturation_warn", 0.8
        )
        assert clean_on_start is not None
        assert monitor_interval is not None
        assert saturation_warn is not None

        store_path.mkdir(parents=True, exist_ok=True)
        if clean_on_start:
            # Remove only this tap's own per-port subdirectories, never
            # store_path itself (which may be the instance's run folder).
            for port in s_ports:
                shutil.rmtree(store_path / port, ignore_errors=True)
        logger.info(
            "tap recording %d timeline(s) %s to %s",
            len(s_ports),
            s_ports,
            store_path,
        )

        metrics = {p: PortMetrics(p) for p in s_ports}
        errors: Dict[str, BaseException] = {}
        monitor = BackpressureMonitor(
            metrics, monitor_interval, saturation_warn
        )
        monitor.start()

        threads: List[threading.Thread] = [
            threading.Thread(
                target=worker,
                args=(
                    instance,
                    p,
                    ids_names[p],
                    store_path,
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

        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"tap timeline(s) failed: {msg}")
            raise RuntimeError(f"tap timeline(s) failed: {msg}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

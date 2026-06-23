"""Tap recorder actor for MUSCLE3.

A terminal (sink-only) actor that taps onto an arbitrary number of independent
*timelines* and records every message it receives to disk, one DBEntry per
message.

The dynamic-port / thread-per-timeline / backpressure machinery is shared with
the distill recorder and lives in :mod:`imas_muscle3.actors._tap_base`; this
module only supplies the per-message *recorder*: each received message is
written to its own ``imas:hdf5?path=<store_path>/<port>/<seq>`` DBEntry,
queryable afterwards with IMAS-Python. ``store_path`` defaults to the
instance's run folder.

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
from pathlib import Path

from imas import DBEntry, IDSFactory
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from libmuscle import Instance, InstanceFlags, Message

from imas_muscle3.actors._tap_base import (
    BackpressureMonitor,
    PortMetrics,
    connected_s_ports,
    ids_name_from_port,
    precompute_ids_metadata,
    serve_timelines,
)
from imas_muscle3.utils import get_setting_optional

# Re-exported for backwards compatibility / tests; they now live in _tap_base.
__all__ = [
    "BackpressureMonitor",
    "PortMetrics",
    "ids_name_from_port",
    "precompute_ids_metadata",
    "record_message",
]

logger = logging.getLogger()


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


class RecordHandler:
    """:class:`~imas_muscle3.actors._tap_base.TimelineHandler` that writes one
    DBEntry per message under ``<store_path>/<port>/<seq>``."""

    def __init__(self, store_path: Path, port: str, ids_name: str) -> None:
        self._store_path = store_path
        self._port = port
        self._ids_name = ids_name

    def handle(self, seq: int, msg: Message) -> str:
        return record_message(
            self._store_path, self._port, self._ids_name, msg.data, seq
        )

    def close(self) -> None:  # nothing to release: each message owns its entry
        pass


def main() -> None:
    """MUSCLE3 execution loop for the tap recorder."""
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    while instance.reuse_instance():
        s_ports = connected_s_ports(instance)
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

        errors = serve_timelines(
            s_ports,
            ids_names,
            lambda port, ids_name: RecordHandler(store_path, port, ids_name),
            instance,
            monitor_interval,
            saturation_warn,
        )

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

"""Tap recorder actor for MUSCLE3.

A terminal (sink-only) actor that taps onto an arbitrary number of independent
*timelines* and records every message it receives to disk, one DBEntry per
message.

The dynamic-port / single-threaded-drain / backpressure machinery and the
shared reuse loop live in :mod:`imas_muscle3.actors._tap_base`
(:func:`~imas_muscle3.actors._tap_base.run_recorder`); this module only
supplies the per-message *recorder*: each received message is written to its
own ``imas:hdf5?path=<store_path>/<port>/<seq>`` DBEntry, queryable afterwards
with IMAS-Python. ``store_path`` defaults to the instance's run folder.

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

from pathlib import Path
from typing import List

from imas import DBEntry
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    BackpressureMonitor,
    HandlerFactory,
    PortMetrics,
    RecorderSettings,
    ids_from_message,
    ids_name_from_port,
    precompute_ids_metadata,
    recorder_main,
)

# Re-exported for backwards compatibility / tests; they now live in _tap_base.
__all__ = [
    "BackpressureMonitor",
    "PortMetrics",
    "ids_name_from_port",
    "precompute_ids_metadata",
    "record_message",
]


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
    ids = ids_from_message(ids_name, data)
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


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Build the per-timeline factory (raw recording has no extra settings)."""
    return lambda port, ids_name: RecordHandler(
        settings.store_path, port, ids_name
    )


if __name__ == "__main__":
    recorder_main("tap", _build_factory)

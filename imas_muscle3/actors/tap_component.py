"""Tap recorder actor for MUSCLE3.

A terminal (sink-only) actor that records each received IDS verbatim: one IMAS
DBEntry per *occurrence* (outer-loop iteration), holding that iteration's trace
and re-openable with IMAS-Python. The shared drain, occurrence numbering and
reuse loop live in :mod:`imas_muscle3.actors._tap_base`
(:class:`~imas_muscle3.actors._tap_base.OccurrenceRecorder`); this module only
supplies the DBEntry sink. ``store_path`` defaults to the instance's run
folder.

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
from typing import List, Optional

from imas import DBEntry
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    HandlerFactory,
    OccurrenceRecorder,
    RecorderSettings,
    ids_from_message,
    ids_name_from_port,
    recorder_main,
)

# Re-exported for backwards compatibility / tests.
__all__ = ["ids_name_from_port"]


class DBEntrySink:
    """A :class:`~imas_muscle3.actors._tap_base.Sink` writing one occurrence's
    messages into a single DBEntry, verbatim, as an IMAS time-trace.

    Streamed slices are ``put_slice``'d into the entry; a whole-trace message
    is ``put`` once. Re-open with ``imas.DBEntry("imas:hdf5?path=<base>",
    "r")``.
    """

    def __init__(self, base: Path, ids_name: str) -> None:
        self._uri = f"imas:hdf5?path={base}"
        self._ids_name = ids_name
        self._entry: Optional[DBEntry] = None

    def write(self, msg: Message) -> str:
        ids = ids_from_message(self._ids_name, msg.data)
        if self._entry is None:
            self._entry = DBEntry(self._uri, "w")
        if (
            len(ids.time) > 1
            or ids.ids_properties.homogeneous_time == IDS_TIME_MODE_INDEPENDENT
        ):
            self._entry.put(ids)
        else:
            self._entry.put_slice(ids)
        return self._uri

    def close(self) -> None:
        if self._entry is not None:
            self._entry.close()
            self._entry = None


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Raw recording has no extra settings; one DBEntry sink per occurrence."""
    return lambda port, ids_name: OccurrenceRecorder(
        settings.store_path / port, ids_name, DBEntrySink
    )


if __name__ == "__main__":
    recorder_main("tap", _build_factory)

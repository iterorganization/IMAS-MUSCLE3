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
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    HandlerFactory,
    OccurrenceRecorder,
    RecorderSettings,
    ids_from_message,
    ids_name_from_port,
    recorder_main,
)
from imas_muscle3.utils import get_setting_optional

# Re-exported for backwards compatibility / tests.
__all__ = ["ids_name_from_port"]


def _put_ids(entry: DBEntry, ids: IDSToplevel) -> None:
    """``put`` a full / time-independent IDS, else ``put_slice`` one slice."""
    if (
        len(ids.time) > 1
        or ids.ids_properties.homogeneous_time == IDS_TIME_MODE_INDEPENDENT
    ):
        entry.put(ids)
    else:
        entry.put_slice(ids)


class DBEntrySink:
    """A :class:`~imas_muscle3.actors._tap_base.Sink` writing one occurrence's
    messages to disk as IMAS, re-openable with ``imas.DBEntry``.

    Default: append every message into one DBEntry (``<base>``), building the
    occurrence's time-trace. With ``per_message``, each message is instead its
    own complete DBEntry (``<base>_<seq>``), written and closed immediately so
    results can be read back while the run is still going.
    """

    def __init__(
        self, base: Path, ids_name: str, per_message: bool = False
    ) -> None:
        self._base = base
        self._ids_name = ids_name
        self._per_message = per_message
        self._entry: Optional[DBEntry] = None
        self._seq = 0

    def write(self, msg: Message) -> str:
        ids = ids_from_message(self._ids_name, msg.data)
        if self._per_message:
            uri = f"imas:hdf5?path={self._base}_{self._seq:08d}"
            with DBEntry(uri, "w") as entry:
                _put_ids(entry, ids)
            self._seq += 1
            return uri
        if self._entry is None:
            self._entry = DBEntry(f"imas:hdf5?path={self._base}", "w")
        _put_ids(self._entry, ids)
        return f"imas:hdf5?path={self._base}"

    def close(self) -> None:
        if self._entry is not None:
            self._entry.close()
            self._entry = None


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """One DBEntry sink per occurrence (or per message, if ``per_message``)."""
    per_message = bool(get_setting_optional(instance, "per_message", False))
    return lambda port, ids_name: OccurrenceRecorder(
        settings.store_path / port,
        ids_name,
        lambda base, name: DBEntrySink(base, name, per_message),
    )


if __name__ == "__main__":
    recorder_main("tap", _build_factory)

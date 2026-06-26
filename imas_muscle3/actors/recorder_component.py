"""Recorder actor for MUSCLE3.

A terminal (sink-only) actor that taps the live traffic of a running workflow
and writes each connected ``S`` port to disk, without disturbing the coupling.
The ``format`` setting picks what is written, one store per *occurrence*
(outer-loop iteration):

- ``imas`` (default): each IDS verbatim as an IMAS DBEntry, re-openable with
  imas-python. With ``per_message``, each message is its own DBEntry, readable
  while the run is going.
- ``raw``: each MUSCLE3 message (frame + undecoded payload) as a msgpack
  stream, no IMAS decode — the cheapest, most faithful capture.
- ``distill``: each IDS reduced to compact arrays in a live, append-only Zarr
  store a viewer can plot.

The drain, occurrence numbering and reuse loop live in :mod:`._tap_base`.

Example yMMSL (yMMSL v0.2)::

    components:
      rec:
        implementation: recorder
        ports:
          s: [equilibrium_in, core_profiles_in]
    settings:
      rec.format: distill
    implementations:
      recorder:
        executable: python
        args: -u -m imas_muscle3.actors.recorder_component
"""

from functools import partial
from pathlib import Path
from typing import BinaryIO, Optional

import msgpack  # type: ignore[import-untyped]
from imas import DBEntry
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    RecorderSettings,
    SinkFactory,
    ids_from_message,
    recorder_main,
)
from imas_muscle3.utils import get_setting_optional


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
    messages as IMAS, re-openable with ``imas.DBEntry``.

    Default: append every message into one DBEntry (``<base>``). With
    ``per_message``, each message is its own complete DBEntry
    (``<base>_<seq>``), written and closed immediately so results can be read
    back while the run is still going.
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


class RawSink:
    """A :class:`~imas_muscle3.actors._tap_base.Sink` writing each MUSCLE3
    message as a msgpack record to one stream per occurrence
    (``<base>.msgpack``).

    Each record is a map ``{"t", "next_t", "data"}`` — the frame plus the
    undecoded wire payload — flushed per message so a reader can tail it live.
    """

    def __init__(self, base: Path, ids_name: str) -> None:
        self._path = base.with_suffix(".msgpack")
        self._fh: Optional[BinaryIO] = None

    def write(self, msg: Message) -> str:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "wb")
        record = {
            "t": msg.timestamp,
            "next_t": msg.next_timestamp,
            "data": bytes(msg.data),
        }
        self._fh.write(msgpack.packb(record, use_bin_type=True))
        self._fh.flush()
        return str(self._path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _build_sink_factory(
    instance: Instance, settings: RecorderSettings
) -> SinkFactory:
    """Pick the per-occurrence sink class from the ``format`` setting."""
    fmt = get_setting_optional(instance, "format", "imas")
    if fmt == "raw":
        return RawSink
    if fmt == "imas":
        per_message = bool(
            get_setting_optional(instance, "per_message", False)
        )
        return partial(DBEntrySink, per_message=per_message)
    if fmt == "distill":
        # Lazy: only the distill format pulls in xarray/zarr/the distiller.
        from imas_muscle3.distill.sink import build_distill_sink_factory

        return build_distill_sink_factory(instance)
    raise ValueError(
        f"unknown recorder format '{fmt}'; use imas, raw or distill."
    )


if __name__ == "__main__":
    recorder_main("recorder", _build_sink_factory)

"""Raw recorder actor for MUSCLE3.

A terminal (sink-only) actor that captures each received MUSCLE3 message
verbatim — its frame (``timestamp``, ``next_timestamp``) plus the undecoded
payload — to one msgpack stream per *occurrence* (outer-loop iteration),
``<store_path>/<port>/<NNNN>.msgpack``. It is the cheapest, most faithful
capture (no IMAS deserialize/serialize round-trip) and is flushed per message,
so it can be tailed while the run is going.

Each record is a msgpack map ``{"t", "next_t", "data"}``; read them back as a
stream with :class:`msgpack.Unpacker`. ``data`` is the IMAS-serialized IDS
payload — replay with ``IDSFactory().new(<ids>).deserialize(rec["data"])`` (the
IDS is the port's name).

The shared drain, occurrence numbering and reuse loop live in
:mod:`imas_muscle3.actors._tap_base`; this module only supplies the raw sink.
``store_path`` defaults to the instance's run folder.

Example yMMSL (yMMSL v0.2)::

    components:
      raw:
        implementation: raw_component
        ports:
          s: [equilibrium_in, core_profiles_in]
    implementations:
      raw_component:
        executable: python
        args: -u -m imas_muscle3.actors.raw_component
"""

from pathlib import Path
from typing import BinaryIO, List, Optional

import msgpack  # type: ignore[import-untyped]
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    HandlerFactory,
    OccurrenceRecorder,
    RecorderSettings,
    recorder_main,
)


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


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Raw capture has no extra settings; one msgpack stream per occurrence."""
    return lambda port, ids_name: OccurrenceRecorder(
        settings.store_path / port, ids_name, RawSink
    )


if __name__ == "__main__":
    recorder_main("raw", _build_factory)

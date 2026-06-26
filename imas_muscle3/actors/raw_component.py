"""Raw recorder actor for MUSCLE3.

A terminal (sink-only) actor that captures each received message's *wire bytes*
verbatim — no IMAS decode at all — to one length-framed file per *occurrence*
(outer-loop iteration), ``<store_path>/<port>/<NNNN>.raw``. It is the cheapest,
most faithful capture (no deserialize/serialize round-trip) and is flushed per
message, so it can be tailed while the run is going. Replay each frame with
``IDSFactory().new(<ids>).deserialize(bytes)`` (the IDS is the port's name).

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

from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    HandlerFactory,
    OccurrenceRecorder,
    RecorderSettings,
    recorder_main,
)


class RawSink:
    """A :class:`~imas_muscle3.actors._tap_base.Sink` writing each message's
    raw wire bytes to one length-framed file per occurrence (``<base>.raw``).

    Each frame is an 8-byte big-endian length followed by the serialized IDS;
    flushed per message so a reader can tail it live.
    """

    def __init__(self, base: Path, ids_name: str) -> None:
        self._path = base.with_suffix(".raw")
        self._fh: Optional[BinaryIO] = None

    def write(self, msg: Message) -> str:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "wb")
        data = bytes(msg.data)
        self._fh.write(len(data).to_bytes(8, "big"))
        self._fh.write(data)
        self._fh.flush()
        return str(self._path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Raw capture has no extra settings; one raw file per occurrence."""
    return lambda port, ids_name: OccurrenceRecorder(
        settings.store_path / port, ids_name, RawSink
    )


if __name__ == "__main__":
    recorder_main("raw", _build_factory)

"""Recorder base class: one instance per connected port, extracting each
received IDS via the shared config logic (see :mod:`.collection`) and
writing it to disk, one store per outer-loop iteration, rolling to a new
occurrence on a stream restart (a message with no ``next_timestamp``, or
time stepping backwards). The on-disk format is left to a subclass, e.g.
:class:`~imas_muscle3.recorder.zarr_recorder.ZarrRecorder`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import xarray as xr
from imas.ids_toplevel import IDSToplevel
from libmuscle import Message

from imas_muscle3.utils import ids_from_message

#: Maps one received IDS to ``name -> Dataset``; every dataset carries a
#: ``time`` dimension (one instant, or a whole trace) to append along.
ExtractFn = Callable[[IDSToplevel], Dict[str, xr.Dataset]]


class Recorder(ABC):
    """Extracts and writes one port's timeline to disk."""

    def __init__(
        self, store_dir: Path, ids_name: str, extract: ExtractFn, profile: str
    ) -> None:
        self._store_dir = store_dir
        self._ids_name = ids_name
        self._extract = extract
        self._profile = profile
        self._occurrence = 0
        self._last_time: Optional[float] = None
        self._prev_ended = False
        self._is_open = False

    def handle(self, msg: Message) -> Tuple[str, Dict[str, xr.Dataset]]:
        """Extract and write one message. Returns a short detail to log and
        the datasets it extracted, for the caller's live state."""
        restarted = self._prev_ended or (
            self._last_time is not None and msg.timestamp < self._last_time
        )
        if self._is_open and restarted:
            self._close_occurrence()
            self._occurrence += 1
            self._is_open = False
        if not self._is_open:
            self._open_occurrence(self._store_dir / f"{self._occurrence:04d}")
            self._is_open = True

        ids = ids_from_message(self._ids_name, msg.data)
        datasets = self._extract(ids)
        detail = self._write(datasets)
        self._last_time = msg.timestamp
        self._prev_ended = msg.next_timestamp is None
        return detail, datasets

    def close(self) -> None:
        """Finalize the currently open occurrence, if any (an empty
        timeline never opened one)."""
        if self._is_open:
            self._close_occurrence()

    @abstractmethod
    def _open_occurrence(self, base: Path) -> None:
        """Open a fresh store at ``base`` (no suffix) for a new occurrence."""

    @abstractmethod
    def _write(self, datasets: Dict[str, xr.Dataset]) -> str:
        """Write one message's extracted datasets; return a short detail."""

    @abstractmethod
    def _close_occurrence(self) -> None:
        """Finalize the currently open occurrence's store."""


#: Builds a Recorder for one port's store dir, IDS name, the shared extract
#: function, and the snapshotted config path (for provenance stamping).
RecorderFactory = Callable[[Path, str, ExtractFn, str], "Recorder"]

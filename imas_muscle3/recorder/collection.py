"""Owns the per-port recorders and live state for one recorder actor run:
loads the shared extract/State config, snapshots it next to the data, and
routes each received message to its port's :class:`~.base.Recorder`
(writing to disk) and :class:`LiveState` (in-memory, a building block for
optional in-actor plotting).
"""

import logging
import runpy
import shutil
from pathlib import Path
from typing import Dict

import xarray as xr
from imas.ids_toplevel import IDSToplevel
from libmuscle import Message

from imas_muscle3.recorder.base import (
    ExtractFn,
    Recorder,
    RecorderFactory,
    RecorderState,
)
from imas_muscle3.recorder.zarr_recorder import _combine
from imas_muscle3.visualization.base_state import BaseState

logger = logging.getLogger()


def load_extract_config(config_path: str) -> ExtractFn:
    """Load the extraction logic from a config file: either a callable
    ``extract(ids) -> dict[str, Dataset]``, or a visualization ``State``
    class (a plot file), each message then going through a fresh instance."""
    namespace = runpy.run_path(config_path)
    extract = namespace.get("extract")
    if extract is not None and callable(extract):
        return extract
    state_class = namespace.get("State")
    if state_class is not None:
        if isinstance(state_class, type) and issubclass(
            state_class, BaseState
        ):

            def extract_via_state(ids: IDSToplevel) -> Dict[str, xr.Dataset]:
                state = state_class({})
                state.extract(ids)
                return dict(state.data)

            return extract_via_state
    raise NameError(
        f"{config_path} must define a callable 'extract(ids)' returning a "
        f"mapping of name -> xarray.Dataset, or a 'State' class inheriting "
        f"from BaseState."
    )


def snapshot_config(config: Path, store_path: Path) -> Path:
    """Copy the config file next to the data, so a viewer can plot the
    stores with the exact code that produced them even after the original
    is edited. Returns the copy (the original if copying failed)."""
    snapshot = store_path / config.name
    try:
        if snapshot.resolve() != config.resolve():
            shutil.copy2(config, snapshot)
        return snapshot
    except OSError:
        logger.warning(
            "could not snapshot config %s to %s",
            config,
            snapshot,
            exc_info=True,
        )
        return config


class LiveState:
    """Accumulates one port's extracted datasets in memory as they arrive --
    live-tailable like the visualization actor's ``State``. A building
    block for optional in-actor plotting (not wired up to a server yet)."""

    def __init__(self) -> None:
        self.data: Dict[str, xr.Dataset] = {}

    def update(self, datasets: Dict[str, xr.Dataset]) -> None:
        for name, ds in datasets.items():
            self.data[name] = (
                _combine([self.data[name], ds]) if name in self.data else ds
            )


class RecorderCollection:
    """One :class:`~.base.Recorder` and one :class:`LiveState` per
    connected port, sharing a single config file."""

    def __init__(
        self,
        store_path: Path,
        config: Path,
        ids_names: Dict[str, str],
        make_recorder: RecorderFactory,
    ) -> None:
        self.extract = load_extract_config(str(config))
        self.config_snapshot = snapshot_config(config, store_path)
        self.live_state: Dict[str, LiveState] = {
            port: LiveState() for port in ids_names
        }
        self._recorders: Dict[str, Recorder] = {
            port: make_recorder(
                store_path / port,
                ids_name,
                self.extract,
                str(self.config_snapshot),
            )
            for port, ids_name in ids_names.items()
        }

    def handle(self, port: str, msg: Message) -> str:
        """Write ``msg`` via ``port``'s recorder and fold it into that
        port's live state; returns a short detail to log."""
        detail, datasets = self._recorders[port].handle(msg)
        self.live_state[port].update(datasets)
        return detail

    def close(self) -> None:
        for recorder in self._recorders.values():
            recorder.close()

    def get_state(self) -> Dict[str, RecorderState]:
        """Every port's :class:`~.base.Recorder` bookkeeping, for a
        checkpoint. Live state is excluded: it's an in-memory-only building
        block for future plotting, not something a resume needs to restore."""
        return {port: rec.get_state() for port, rec in self._recorders.items()}

    def restore_state(self, state: Dict[str, RecorderState]) -> None:
        """Resume every port's recorder from a previous :meth:`get_state`."""
        for port, port_state in state.items():
            self._recorders[port].restore_state(port_state)

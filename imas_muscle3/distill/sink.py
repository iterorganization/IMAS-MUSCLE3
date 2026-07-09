"""The distill :class:`Sink`: extract plot-ready datasets from each received
IDS and append them to one Zarr store per occurrence."""

import logging
import runpy
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import xarray as xr
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    RecorderSettings,
    SinkFactory,
    ids_from_message,
)
from imas_muscle3.distill.zarr_sink import ZarrSink, write_root_attrs

logger = logging.getLogger()

#: Maps one received IDS to ``name -> Dataset``; every dataset carries a
#: ``time`` dimension (one instant, or a whole trace) to append along.
ExtractFn = Callable[[IDSToplevel], Dict[str, xr.Dataset]]


def load_extract_config(config_path: str) -> ExtractFn:
    """Load the extraction logic from a config file.

    The file defines either a callable ``extract(ids) -> dict[str, Dataset]``,
    or a visualization ``State`` class (a
    :class:`~imas_muscle3.visualization.base_state.BaseState` subclass, i.e. a
    live-actor plot file) whose ``extract`` method is wrapped: each message is
    fed to a fresh instance and its accumulated ``data`` datasets are
    recorded. Sharing one plot file between this recorder and a viewer's
    ``Plotter`` keeps the stored quantities and the plots that read them in
    lockstep.
    """
    namespace = runpy.run_path(config_path)
    extract = namespace.get("extract")
    if extract is not None and callable(extract):
        return extract
    state_class = namespace.get("State")
    if state_class is not None:
        from imas_muscle3.visualization.base_state import BaseState

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


class DistillSink:
    """A :class:`~imas_muscle3.actors._tap_base.Sink` appending each
    message's extracted datasets to one occurrence's Zarr store
    (``<base>.zarr``)."""

    def __init__(
        self, base: Path, ids_name: str, extract: ExtractFn, profile: str
    ) -> None:
        self._store = base.with_suffix(".zarr")
        self._ids_name = ids_name
        self._extract = extract
        self._profile = profile
        self._zarr = ZarrSink(self._store)

    def write(self, msg: Message) -> str:
        ids = ids_from_message(self._ids_name, msg.data)
        datasets = self._extract(ids)
        for name, ds in datasets.items():
            self._zarr.append(name, ds)
        return f"{len(datasets)} dataset(s)"

    def close(self) -> None:
        self._zarr.close()
        # Stamp the occurrence index + config reference so a viewer can group
        # stores and load the matching plots.
        write_root_attrs(
            self._store,
            {
                "occurrence": int(self._store.stem),
                "distill_profile": str(Path(self._profile).resolve()),
            },
        )


def build_distill_sink_factory(
    instance: Instance, settings: RecorderSettings
) -> SinkFactory:
    """Read the required ``config`` setting and bind it to the sink."""
    config = str(instance.get_setting("config", "str"))
    extract = load_extract_config(config)
    logger.info("distilling with config=%s", config)
    return partial(DistillSink, extract=extract, profile=config)

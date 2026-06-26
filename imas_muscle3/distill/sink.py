"""The distill :class:`Sink` and its factory, for the recorder's ``distill``
format. Distills each message into compact arrays and appends them along
``time`` to one Zarr store per occurrence — small, self-describing, and
live-tailable as the run writes it."""

import logging
import runpy
from pathlib import Path
from typing import List, Optional

from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import (
    HandlerFactory,
    OccurrenceRecorder,
    RecorderSettings,
    ids_from_message,
)
from imas_muscle3.distill.distiller import Distiller, ExtractFn
from imas_muscle3.distill.zarr_sink import ZarrSink, write_root_attrs
from imas_muscle3.utils import get_setting_optional

logger = logging.getLogger()


def load_extract_config(config_path: str) -> ExtractFn:
    """Load an ``extract(ids) -> dict[str, Dataset]`` callable from a file."""
    namespace = runpy.run_path(config_path)
    extract = namespace.get("extract")
    if extract is None or not callable(extract):
        raise NameError(
            f"{config_path} must define a callable 'extract(ids)' returning "
            f"a mapping of name -> single-time xarray.Dataset."
        )
    return extract


class DistillSink:
    """:class:`~imas_muscle3.actors._tap_base.Sink` that distills each message
    and appends it to one occurrence's Zarr store (``<base>.zarr``)."""

    def __init__(
        self,
        base: Path,
        ids_name: str,
        distiller: Distiller,
        profile: Optional[str] = None,
    ) -> None:
        self._store = base.with_suffix(".zarr")
        self._ids_name = ids_name
        self._distiller = distiller
        self._profile = profile
        self._zarr = ZarrSink(self._store)

    def write(self, msg: Message) -> str:
        ids = ids_from_message(self._ids_name, msg.data)
        datasets = self._distiller.distill(ids)
        for full_path, ds in datasets.items():
            self._zarr.append(full_path, ds)
        return f"{len(datasets)} var(s)"

    def close(self) -> None:
        self._zarr.close()
        # Stamp the occurrence index (+ profile) so a viewer can group stores
        # and load the matching bespoke plots.
        meta: dict = {"occurrence": int(self._store.stem)}
        if self._profile:
            meta["distill_profile"] = str(Path(self._profile).resolve())
        write_root_attrs(self._store, meta)


def build_distill_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Read distill settings (``auto``, ``config``) and build the factory."""
    auto = get_setting_optional(instance, "auto", True)
    config = get_setting_optional(instance, "config")
    assert auto is not None
    extract = load_extract_config(config) if config else None
    logger.info("distilling with auto=%s, config=%s", auto, config or "-")

    def factory(port: str, ids_name: str) -> OccurrenceRecorder:
        # A fresh Distiller per timeline keeps its discovery cache local, so
        # the per-sender workers never share mutable state.
        distiller = Distiller(auto=auto, extract=extract)
        return OccurrenceRecorder(
            settings.store_path / port,
            ids_name,
            lambda base, name: DistillSink(base, name, distiller, config),
        )

    return factory

"""The distill :class:`Sink` and its factory, for the recorder's ``distill``
format. Distills each message into compact arrays and appends them along
``time`` to one Zarr store per occurrence — small, self-describing, and
live-tailable as the run writes it."""

import logging
import runpy
from functools import partial
from pathlib import Path
from typing import Optional

from libmuscle import Instance, Message

from imas_muscle3.actors._tap_base import SinkFactory, ids_from_message
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
    and appends it to one occurrence's Zarr store (``<base>.zarr``). Owns a
    fresh :class:`Distiller`, so its discovery cache is local to the occurrence
    and the per-sender worker threads share no mutable state."""

    def __init__(
        self,
        base: Path,
        ids_name: str,
        auto: bool = True,
        extract: Optional[ExtractFn] = None,
        profile: Optional[str] = None,
    ) -> None:
        self._store = base.with_suffix(".zarr")
        self._ids_name = ids_name
        self._distiller = Distiller(auto=auto, extract=extract)
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


def build_distill_sink_factory(instance: Instance) -> SinkFactory:
    """Read distill settings (``auto``, ``config``) and bind them to the
    sink."""
    auto = get_setting_optional(instance, "auto", True)
    config = get_setting_optional(instance, "config")
    assert auto is not None
    extract = load_extract_config(config) if config else None
    logger.info("distilling with auto=%s, config=%s", auto, config or "-")
    return partial(DistillSink, auto=auto, extract=extract, profile=config)

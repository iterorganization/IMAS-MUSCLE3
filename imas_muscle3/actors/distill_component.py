"""Distill recorder actor for MUSCLE3.

A terminal (sink-only) actor that, instead of storing each raw IDS (see
:mod:`tap_component`), *distills* every message into compact scalars / profiles
/ maps and appends them along ``time`` to a Zarr store — one store per
*occurrence* (outer-loop iteration), so iterations sit side by side. The result
is a small, self-describing, append-only dataset a viewer can plot live as the
run writes it or open afterwards.

The shared drain, occurrence numbering and reuse loop live in
:mod:`imas_muscle3.actors._tap_base`
(:class:`~imas_muscle3.actors._tap_base.OccurrenceRecorder`); this module only
supplies the distill-and-append sink and its settings.

Settings (all optional):

- ``store_path``: where the per-port output goes (default: the instance's run
  folder).
- ``auto`` (default ``true``): auto-discover and record every time-dependent
  0D/1D/2D ``FLT`` quantity in each IDS.
- ``config``: path to a Python file defining ``extract(ids) -> dict[str,
  xarray.Dataset]`` for derived/geometric quantities; recorded in addition to
  (or, with ``auto: false``, instead of) the auto-discovered ones.

Example yMMSL (yMMSL v0.2)::

    components:
      distill:
        implementation: distill_component
        ports:
          s: [equilibrium_in, core_profiles_in]
    settings:
      distill.store_path: /scratch/distill_store
    implementations:
      distill_component:
        executable: python
        args: -u -m imas_muscle3.actors.distill_component
"""

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
    recorder_main,
)
from imas_muscle3.distill import Distiller, ZarrSink
from imas_muscle3.distill.distiller import ExtractFn
from imas_muscle3.distill.zarr_sink import write_root_attrs
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


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Read distill-specific settings and build the per-timeline factory."""
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


if __name__ == "__main__":
    recorder_main("distill", _build_factory)

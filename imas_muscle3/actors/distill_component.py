"""Distill recorder actor for MUSCLE3.

A terminal (sink-only) tap that, instead of recording each raw IDS (see
:mod:`tap_component`), *distills* every message into compact scalars / profiles
/ maps and appends them along ``time`` to a Zarr store. The result is a small,
self-describing, append-only dataset that a viewer can plot live as the run
writes it or open afterwards.

The output is split into one *occurrence* per outer-loop iteration —
``<store_path>/<port>/<NNNN>.zarr`` — derived purely from the message stream
(see :class:`DistillHandler` and ``reuse_and_close.md``), so a driven workflow
that re-runs the same time grid each iteration lands each iteration in its own
occurrence with no extra wiring.

The dynamic-port / single-threaded-drain / backpressure machinery and the
shared reuse loop live in :mod:`imas_muscle3.actors._tap_base`
(:func:`~imas_muscle3.actors._tap_base.run_recorder`); this module only
supplies the distill-and-append handler and the distill-specific settings.

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


class DistillHandler:
    """:class:`~imas_muscle3.actors._tap_base.TimelineHandler` that distills
    each message and appends it to the current *occurrence*'s Zarr store,
    rolling to a new occurrence whenever the timeline restarts.

    An occurrence is one contiguous, time-ordered run of messages. A new one
    begins when the previous message ended a stream (``next_timestamp is
    None``) or simulation time steps backwards — i.e. an outer-loop iteration
    re-running the same time grid. Occurrences are written side by side as
    ``<store_dir>/<NNNN>.zarr`` for the viewer to compare — derived purely from
    the message stream, so it needs no per-iteration trigger or wiring.
    """

    def __init__(
        self,
        store_dir: Path,
        ids_name: str,
        distiller: Distiller,
        profile: Optional[str] = None,
    ) -> None:
        self._store_dir = store_dir
        self._ids_name = ids_name
        self._distiller = distiller
        self._profile = profile
        self._occurrence = 0
        self._sink: Optional[ZarrSink] = None
        self._last_time: Optional[float] = None
        self._prev_ended = False

    def _store(self) -> Path:
        return self._store_dir / f"{self._occurrence:04d}.zarr"

    def _close_current(self) -> None:
        if self._sink is None:
            return
        self._sink.close()
        # Stamp the occurrence index (+ profile) so the viewer can group and
        # load the matching bespoke plots.
        meta: dict = {"occurrence": self._occurrence}
        if self._profile:
            meta["distill_profile"] = str(Path(self._profile).resolve())
        write_root_attrs(self._store(), meta)
        self._sink = None

    def handle(self, seq: int, msg: Message) -> str:
        restarted = self._prev_ended or (
            self._last_time is not None and msg.timestamp < self._last_time
        )
        if self._sink is not None and restarted:
            self._close_current()
            self._occurrence += 1
        if self._sink is None:
            self._sink = ZarrSink(self._store())
        ids = ids_from_message(self._ids_name, msg.data)
        datasets = self._distiller.distill(ids)
        for full_path, ds in datasets.items():
            self._sink.append(full_path, ds)
        self._last_time = msg.timestamp
        self._prev_ended = msg.next_timestamp is None
        return f"occ {self._occurrence:04d}, {len(datasets)} var(s)"

    def close(self) -> None:
        self._close_current()


def _build_factory(
    instance: Instance, settings: RecorderSettings, s_ports: List[str]
) -> HandlerFactory:
    """Read distill-specific settings and build the per-timeline factory."""
    auto = get_setting_optional(instance, "auto", True)
    config = get_setting_optional(instance, "config")
    assert auto is not None
    extract = load_extract_config(config) if config else None
    logger.info("distilling with auto=%s, config=%s", auto, config or "-")

    # A fresh Distiller per timeline keeps its discovery cache thread-local, so
    # the per-port workers never share mutable state.
    return lambda port, ids_name: DistillHandler(
        settings.store_path / port,
        ids_name,
        Distiller(auto=auto, extract=extract),
        config,
    )


if __name__ == "__main__":
    recorder_main("distill", _build_factory)

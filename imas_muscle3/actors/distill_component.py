"""Distill recorder actor for MUSCLE3.

A terminal (sink-only) tap that, instead of recording each raw IDS (see
:mod:`tap_component`), *distills* every message into compact scalars / profiles
/ maps and appends them along ``time`` to a Zarr store — one store per timeline
(``<store_path>/<port>.zarr``), a group per distilled variable. The result is a
small, self-describing, append-only dataset that a viewer can plot live as the
run writes it or open afterwards.

All the dynamic-port / thread-per-timeline / backpressure machinery is shared
with the tap recorder via :mod:`imas_muscle3.actors._tap_base`; this module only
supplies the distill-and-append handler.

Settings (all optional):

- ``store_path``: where the per-port Zarr stores go (default: the instance's
  run folder).
- ``auto`` (default ``true``): auto-discover and record every time-dependent
  0D/1D/2D ``FLT`` quantity in each IDS.
- ``config``: path to a Python file defining ``extract(ids) -> dict[str,
  xarray.Dataset]`` for derived/geometric quantities; recorded in addition to
  (or, with ``auto: false``, instead of) the auto-discovered ones.
- ``clean_on_start`` (default ``true``): remove this tap's own per-port stores
  before recording.
- ``monitor_interval`` / ``saturation_warn``: backpressure logging knobs.

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
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from imas import IDSFactory
from libmuscle import Instance, InstanceFlags, Message

from imas_muscle3.actors._tap_base import (
    connected_s_ports,
    ids_name_from_port,
    serve_timelines,
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
    each message and appends the result to this timeline's Zarr store."""

    def __init__(
        self, store: Path, ids_name: str, distiller: Distiller
    ) -> None:
        self._ids_name = ids_name
        self._distiller = distiller
        self._sink = ZarrSink(store)

    def handle(self, seq: int, msg: Message) -> str:
        ids = IDSFactory().new(self._ids_name)
        ids.deserialize(msg.data)
        datasets = self._distiller.distill(ids)
        for full_path, ds in datasets.items():
            self._sink.append(full_path, ds)
        return f"{len(datasets)} var(s)"

    def close(self) -> None:
        self._sink.close()


def _make_distiller(auto: bool, extract: Optional[ExtractFn]) -> Distiller:
    # A fresh Distiller per timeline keeps its discovery cache thread-local,
    # so the per-port workers never share mutable state.
    return Distiller(auto=auto, extract=extract)


def _read_settings(instance: Instance) -> SimpleNamespace:
    """Read the actor's settings once (they are constant across reuses)."""
    store_path_setting = get_setting_optional(instance, "store_path")
    store_path = (
        Path(store_path_setting)
        if store_path_setting is not None
        else Path.cwd()
    )
    auto = get_setting_optional(instance, "auto", True)
    config = get_setting_optional(instance, "config")
    clean_on_start = get_setting_optional(instance, "clean_on_start", True)
    monitor_interval = get_setting_optional(instance, "monitor_interval", 5.0)
    saturation_warn = get_setting_optional(instance, "saturation_warn", 0.8)
    assert auto is not None
    assert clean_on_start is not None
    assert monitor_interval is not None
    assert saturation_warn is not None
    return SimpleNamespace(
        store_path=store_path,
        auto=auto,
        config=config,
        extract=load_extract_config(config) if config else None,
        clean_on_start=clean_on_start,
        monitor_interval=monitor_interval,
        saturation_warn=saturation_warn,
    )


def main() -> None:
    """MUSCLE3 execution loop for the distill recorder.

    Each reuse (one F_INIT loop / Picard iteration in a driven workflow) is
    recorded as its own *occurrence*: per port ``<store_path>/<port>/<NNNN>.zarr``,
    NNNN being the reuse index. Whatever a timeline carries that reuse — a single
    slice, a stream of slices, or a whole trace — lands in that occurrence's
    store. This mirrors IDS occurrences: successive versions of the same
    quantity, here one per loop, side by side for the viewer to compare.
    """
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    occurrence = 0
    setup: Optional[SimpleNamespace] = None
    while instance.reuse_instance():
        s_ports = connected_s_ports(instance)
        if not s_ports:
            logger.warning(
                "distill recorder has no connected S ports; nothing to record."
            )
            break
        ids_names = {p: ids_name_from_port(p) for p in s_ports}

        if setup is None:
            setup = _read_settings(instance)
            setup.store_path.mkdir(parents=True, exist_ok=True)
            if setup.clean_on_start:
                # Clear this tap's own per-port occurrence dirs once, up front;
                # never store_path itself (it may be the instance run folder).
                for port in s_ports:
                    shutil.rmtree(setup.store_path / port, ignore_errors=True)

        logger.info(
            "distilling occurrence %04d: %d timeline(s) %s to %s "
            "(auto=%s, config=%s)",
            occurrence,
            len(s_ports),
            s_ports,
            setup.store_path,
            setup.auto,
            setup.config or "-",
        )

        def occurrence_store(port: str, occ: int = occurrence) -> Path:
            return setup.store_path / port / f"{occ:04d}.zarr"

        errors = serve_timelines(
            s_ports,
            ids_names,
            lambda port, ids_name: DistillHandler(
                occurrence_store(port),
                ids_name,
                _make_distiller(setup.auto, setup.extract),
            ),
            instance,
            setup.monitor_interval,
            setup.saturation_warn,
        )

        # Record which profile produced each store + the occurrence index, so
        # the viewer can group occurrences and load the matching bespoke plots.
        for port in s_ports:
            meta: dict = {"occurrence": occurrence}
            if setup.config:
                meta["distill_profile"] = str(Path(setup.config).resolve())
            write_root_attrs(occurrence_store(port), meta)

        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"distill timeline(s) failed: {msg}")
            raise RuntimeError(f"distill timeline(s) failed: {msg}")

        occurrence += 1


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

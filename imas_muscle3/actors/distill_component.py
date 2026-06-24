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

All the dynamic-port / single-threaded-drain / backpressure machinery is shared
with the tap recorder via :mod:`imas_muscle3.actors._tap_base`; this module
only supplies the distill-and-append handler.

Settings (all optional):

- ``store_path``: where the per-port output goes (default: the instance's run
  folder).
- ``auto`` (default ``true``): auto-discover and record every time-dependent
  0D/1D/2D ``FLT`` quantity in each IDS.
- ``config``: path to a Python file defining ``extract(ids) -> dict[str,
  xarray.Dataset]`` for derived/geometric quantities; recorded in addition to
  (or, with ``auto: false``, instead of) the auto-discovered ones.
- ``clean_on_start`` (default ``true``): remove this tap's own per-port output
  before recording.
- ``monitor_interval`` / ``saturation_warn``: backpressure logging knobs.

The recorder drains its ports **single-threaded** (round-robin, one outstanding
``receive``) and to the real port close; see
:mod:`imas_muscle3.actors._tap_base` and ``reuse_and_close.md``.

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
        ids = IDSFactory().new(self._ids_name)
        ids.deserialize(msg.data)
        datasets = self._distiller.distill(ids)
        for full_path, ds in datasets.items():
            self._sink.append(full_path, ds)
        self._last_time = msg.timestamp
        self._prev_ended = msg.next_timestamp is None
        return f"occ {self._occurrence:04d}, {len(datasets)} var(s)"

    def close(self) -> None:
        self._close_current()


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

    Each timeline is drained to its end and split into occurrences
    (``<store_path>/<port>/<NNNN>.zarr``) at every stream restart — a
    ``next_timestamp is None`` boundary or a backward time step — so a workflow
    that re-runs the same grid per outer-loop iteration lands each iteration in
    its own occurrence. The occurrence index is derived entirely from the
    message stream, with no extra wiring (see ``reuse_and_close.md``).
    """
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

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
                # Clear this tap's own per-port dirs once, up front; never
                # store_path itself (it may be the instance's run folder).
                for port in s_ports:
                    shutil.rmtree(setup.store_path / port, ignore_errors=True)

        logger.info(
            "distilling %d timeline(s) %s to %s (auto=%s, config=%s)",
            len(s_ports),
            s_ports,
            setup.store_path,
            setup.auto,
            setup.config or "-",
        )

        errors = serve_timelines(
            s_ports,
            ids_names,
            lambda port, ids_name: DistillHandler(
                setup.store_path / port,
                ids_name,
                _make_distiller(setup.auto, setup.extract),
                setup.config,
            ),
            instance,
            setup.monitor_interval,
            setup.saturation_warn,
        )

        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"distill timeline(s) failed: {msg}")
            raise RuntimeError(f"distill timeline(s) failed: {msg}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

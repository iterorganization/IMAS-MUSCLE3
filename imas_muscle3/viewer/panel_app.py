"""Panel app for browsing distilled Zarr stores, plus the dashboard plugin hook.

The :class:`DistillViewer` lets you pick a timeline (store), an IDS (group) and a
variable, and plots it; while a run is still writing, it live-tails by re-opening
the store to pick up appended time steps.

It is surfaced to the generic ``muscle3-dashboard`` through a tiny, duck-typed
contract so the dashboard needs no imas/zarr dependency:

    entry-point group:  ``muscle3_dashboard.run_panels``
    factory:            ``make_panel(run_dir: Path) -> RunPanel | None``
    RunPanel:           ``.title: str`` and ``.view() -> panel Viewable``

The dashboard calls every registered factory with a run directory; a factory
returns ``None`` when it finds nothing it can show (here: no ``*.zarr`` stores),
otherwise a :class:`RunPanel` the dashboard renders as a card/tab. ``view`` is a
thunk so the (possibly heavy) panel is built only when actually shown.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import holoviews as hv
import panel as pn
import param
from panel.viewable import Viewable, Viewer

from imas_muscle3.viewer import store as store_mod
from imas_muscle3.viewer.plots import plot_variable
from imas_muscle3.viewer.profile import ProfileData, load_profile

logger = logging.getLogger(__name__)

hv.extension("bokeh")

#: How often the live tail re-opens the store to pick up appended time steps.
REFRESH_MS = 2000


@dataclass
class RunPanel:
    """A dashboard run-panel: a title and a lazy builder of its view."""

    title: str
    view: Callable[[], Viewable]


class DistillViewer(Viewer):
    """Browse and plot the distilled Zarr stores under one run directory."""

    store = param.Selector(objects=[], doc="Timeline (one Zarr store per port).")
    group = param.Selector(objects=[], doc="Recorded IDS within the store.")
    variable = param.Selector(objects=[], doc="Distilled quantity to plot.")
    time_index = param.Integer(default=0, bounds=(0, 0))
    live = param.Boolean(default=True, label="Live (follow latest)")

    def __init__(self, run_dir: Path, **params: object) -> None:
        super().__init__(**params)
        self.run_dir = Path(run_dir)
        self._ds = None
        self._stores: dict[str, Path] = {}
        self._discover_stores()

    # --- store/group/variable cascade --------------------------------------

    def _discover_stores(self) -> None:
        self._stores = {
            store_mod.store_label(p): p
            for p in store_mod.find_stores(self.run_dir)
        }
        labels = sorted(self._stores)
        self.param.store.objects = labels
        if labels and self.store not in self._stores:
            self.store = labels[0]

    @param.depends("store", watch=True)
    def _on_store(self) -> None:
        store = self._stores.get(self.store)
        groups = store_mod.list_groups(store) if store else []
        self.param.group.objects = groups
        self.group = groups[0] if groups else None

    @param.depends("group", watch=True)
    def _on_group(self) -> None:
        self._reload()

    def _reload(self) -> None:
        """(Re)open the selected group and refresh variable/time options."""
        store = self._stores.get(self.store)
        if not store or not self.group:
            self._ds = None
            self.param.variable.objects = []
            self.variable = None
            return
        try:
            self._ds = store_mod.open_group(store, self.group)
        except Exception:
            logger.warning(
                "failed to open %s/%s", store, self.group, exc_info=True
            )
            return
        variables = store_mod.plottable_variables(self._ds)
        self.param.variable.objects = variables
        if self.variable not in variables:
            self.variable = variables[0] if variables else None
        self._update_time_bounds()

    def _update_time_bounds(self) -> None:
        n = self._ds.sizes.get(store_mod.TIME, 0) if self._ds is not None else 0
        self.param.time_index.bounds = (0, max(0, n - 1))
        if self.live and n:
            self.time_index = n - 1

    def refresh(self) -> None:
        """Live tail: pick up newly appended time steps (and late stores)."""
        if not self._stores:
            self._discover_stores()
            self._on_store()
        elif self._ds is not None:
            self._reload()

    # --- view ---------------------------------------------------------------

    @param.depends("variable", "time_index")
    def _plot(self) -> Viewable:
        if self._ds is None or not self.variable:
            return pn.pane.Markdown("### Waiting for distilled data…")
        return pn.pane.HoloViews(
            plot_variable(self._ds, self.variable, self.time_index),
            sizing_mode="stretch_both",
            min_height=400,
        )

    def __panel__(self) -> Viewable:
        selectors = pn.Row(
            pn.widgets.Select.from_param(self.param.store, name="Timeline"),
            pn.widgets.Select.from_param(self.param.group, name="IDS"),
            pn.widgets.Select.from_param(self.param.variable, name="Variable"),
        )
        time = pn.Row(
            pn.widgets.IntSlider.from_param(
                self.param.time_index, name="Time index"
            ),
            pn.widgets.Checkbox.from_param(self.param.live),
        )
        return pn.Column(
            selectors, time, self._plot, sizing_mode="stretch_width"
        )


class ProfileView(Viewer):
    """Render a workflow's bespoke profile ``plot`` for one F_INIT loop.

    The profile author writes a pure ``plot(data, time_index)``; this view owns
    the interactivity: an *occurrence* selector (which F_INIT loop / Picard
    iteration to show), a time slider over that occurrence's trace, a
    live-follow toggle, and a poll that re-reads the stores so new occurrences
    appear and the latest one's trace grows with the run.
    """

    occurrence = param.Selector(objects=[], doc="F_INIT loop / iteration.")
    time_index = param.Integer(default=0, bounds=(0, 0))
    live = param.Boolean(default=True, label="Live (follow latest)")
    _tick = param.Integer(default=0)

    def __init__(self, run_dir: Path, profile_path: str, **params: object):
        super().__init__(**params)
        self.run_dir = Path(run_dir)
        self._profile = load_profile(profile_path)
        self._data = ProfileData({})
        self._discover_occurrences()

    def _discover_occurrences(self) -> None:
        self._occurrences = store_mod.occurrences(self.run_dir)
        labels = list(self._occurrences)
        self.param.occurrence.objects = labels
        if labels and self.occurrence not in self._occurrences:
            # Default to the latest occurrence (most recent iteration).
            self.occurrence = labels[-1]
        else:
            self._rebuild_data()

    @param.depends("occurrence", watch=True)
    def _rebuild_data(self) -> None:
        self._data = ProfileData(self._occurrences.get(self.occurrence, {}))
        self._update_time_bounds()

    def _update_time_bounds(self) -> None:
        n = self._data.n_times()
        self.param.time_index.bounds = (0, max(0, n - 1))
        if self.live and n:
            self.time_index = n - 1

    def refresh(self) -> None:
        was_latest = (
            not self._occurrences
            or self.occurrence == list(self._occurrences)[-1]
        )
        self._discover_occurrences()
        # While live, follow the newest occurrence as iterations complete.
        if self.live and was_latest and self._occurrences:
            self.occurrence = list(self._occurrences)[-1]
        self._data.reset()
        self._update_time_bounds()
        self._tick += 1  # force a re-render even if nothing else changed

    @param.depends("occurrence", "time_index", "_tick")
    def _view(self) -> Viewable:
        if self._profile.plot is None:
            return pn.pane.Markdown("### Profile defines no `plot`.")
        try:
            return pn.panel(self._profile.plot(self._data, self.time_index))
        except Exception:
            logger.warning("profile plot failed", exc_info=True)
            return pn.pane.Markdown("### Profile `plot` raised; see logs.")

    def __panel__(self) -> Viewable:
        controls = pn.Row(
            pn.widgets.Select.from_param(
                self.param.occurrence, name="F_INIT loop"
            ),
            pn.widgets.IntSlider.from_param(
                self.param.time_index, name="Time index"
            ),
            pn.widgets.Checkbox.from_param(self.param.live),
        )
        return pn.Column(controls, self._view, sizing_mode="stretch_width")


def _profiles_for(run_dir: Path) -> list[str]:
    """Distinct visualization profiles stamped across a run's stores."""
    seen: list[str] = []
    for store in store_mod.find_stores(run_dir):
        profile = store_mod.store_profile(store)
        if profile and profile not in seen:
            seen.append(profile)
    return seen


def _mounted_view(run_dir: Path) -> Viewable:
    """Build the run's view and, in a live session, start its live-tail polls.

    A profile-stamped run shows a tab per profile (its bespoke plots) plus a
    generic "Browse" tab; a plain run shows just the generic browser.
    """
    run_dir = Path(run_dir)
    live = pn.state.curdoc is not None

    def mount(view: object) -> object:
        if live and hasattr(view, "refresh"):
            pn.state.add_periodic_callback(view.refresh, REFRESH_MS)
        return view

    profiles = _profiles_for(run_dir)
    browser = mount(DistillViewer(run_dir))
    if not profiles:
        return browser

    tabs = pn.Tabs(sizing_mode="stretch_width")
    for path in profiles:
        try:
            tabs.append((Path(path).stem, mount(ProfileView(run_dir, path))))
        except Exception:
            logger.warning("could not load profile %s", path, exc_info=True)
    tabs.append(("Browse", browser))
    return tabs


def make_panel(run_dir: Path) -> Optional[RunPanel]:
    """Plugin entry point: a distilled-plots panel, or None if no stores."""
    if not store_mod.find_stores(Path(run_dir)):
        return None
    return RunPanel(
        title="Distilled plots", view=lambda: _mounted_view(run_dir)
    )


def serve(run_dir: Path, port: int = 0, show: bool = True) -> None:
    """Serve the viewer standalone (without the dashboard)."""
    pn.serve(
        lambda: _mounted_view(run_dir),
        port=port,
        show=show,
        title="Distilled plots",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Browse distilled Zarr stores.")
    parser.add_argument("run_dir", type=Path, help="Run directory to scan.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    serve(args.run_dir, port=args.port, show=not args.no_show)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

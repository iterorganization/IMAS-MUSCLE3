"""Panel app for browsing recorded IMAS data, plus the dashboard plugin hook.

The :class:`ImasPlotsViewer` lets you pick a Component (recording instance),
Timeline (port) and Outer loop (occurrence), then an IDS and a set of
variables, and plots them in a grid (same-units variables overlaid). A time
player animates; while a run is still writing it live-tails by re-opening the
store to pick up appended time steps and holds at the latest frame.

It is surfaced to the generic ``muscle3-dashboard`` through a tiny, duck-typed
contract so the dashboard needs no imas/zarr dependency:

    entry-point group:  ``muscle3_dashboard.run_panels``
    factory:            ``make_panel(run_dir: Path) -> RunPanel | None``
    RunPanel:           ``.title: str`` and ``.view() -> panel Viewable``

The dashboard calls every registered factory with a run directory; a factory
returns ``None`` when it finds nothing it can show (here: no ``*.zarr``
stores), otherwise a :class:`RunPanel` the dashboard renders as a card/tab.
``view`` is a thunk so the (heavy) panel is built only when actually shown.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import holoviews as hv
import panel as pn
import param
from panel.viewable import Viewable, Viewer

from imas_muscle3.viewer import store as store_mod
from imas_muscle3.viewer.plots import plot_overlay
from imas_muscle3.viewer.profile import ProfileData, load_profile

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

hv.extension("bokeh")

#: How often the live tail re-opens the store to pick up appended time steps.
REFRESH_MS = 2000

#: How often the time player advances one step while animating.
PLAY_MS = 200

#: Poll cycles with no new time steps before a run counts as finished (so the
#: player may loop instead of holding at the latest frame).
_FINISHED_AFTER_STALE_POLLS = 2

#: Variables preselected per IDS on first load, to land on the quantities the
#: old inline visualization actor showed. Matched as substrings of the
#: distilled (dotted DD path) variable names; missing ones are skipped.
_DEFAULT_VARS = {
    "equilibrium": [
        "global_quantities.ip",
        "global_quantities.beta_tor",
        "boundary.outline",
        "profiles_1d.q",
        "profiles_1d.pressure",
        "profiles_1d.f_df_dpsi",
        "profiles_1d.dpressure_dpsi",
    ],
    "core_profiles": [
        "electrons.density",
        "electrons.temperature",
        "t_i_average",
        "j_total",
        "q",
        "zeff",
    ],
    "pf_active": ["coil.current.data"],
}


@dataclass
class RunPanel:
    """A dashboard run-panel: a title and a lazy builder of its view."""

    title: str
    view: Callable[[], Viewable]


class ImasPlotsViewer(Viewer):
    """Browse and plot recorded IMAS data under one run directory.

    The store identity is split across cascading selectors — **Component** (the
    MUSCLE instance that recorded), **IDS** (the recorded stream) and **Outer
    loop** (the F_INIT occurrence / Picard iteration) — which resolve one Zarr
    store. (There is no separate "timeline" selector: a recorder writes one IDS
    per port, so the port name just duplicated the IDS.) A multi-select of
    variables drives a responsive grid of plots; variables that share units
    (and, for profiles, a coordinate) overlay into a single plot. A time player
    auto-advances, holding at the latest step while the run is still writing
    and only looping once it stops; over-time plots get a moving time marker
    and share one time axis.
    """

    component = param.Selector(
        objects=[], doc="MUSCLE instance that recorded."
    )
    ids = param.Selector(objects=[], doc="Recorded IDS stream.")
    outer_loop = param.Selector(
        objects=[], doc="F_INIT loop / Picard iteration."
    )
    variables = param.ListSelector(
        default=[], objects=[], doc="Quantities to plot (overlaid by units)."
    )
    time_index = param.Integer(default=0, bounds=(0, 0))
    playing = param.Boolean(default=True, label="Animate")
    _struct = param.Integer(default=0, doc="Bumped to rebuild the plot grid.")

    #: Component label for stores not under an instances/<name>/workdir tree.
    _NO_COMPONENT = "(run)"

    def __init__(self, run_dir: Path, **params: object) -> None:
        super().__init__(**params)
        self.run_dir = Path(run_dir)
        self._ds: Optional["xr.Dataset"] = None  # open store/group dataset
        # component -> ids (group) -> outer_loop (occurrence) -> store path
        self._index: dict = {}
        self._writing = True  # assume live until polls show no new steps
        self._stale_polls = 0
        self._last_n = 0
        self._discover()

    # --- store discovery + selector cascade --------------------------------

    def _build_index(self) -> dict:
        index: dict = {}
        for store in store_mod.find_stores(self.run_dir):
            comp = store_mod.store_instance(store) or self._NO_COMPONENT
            occ = store_mod.store_occurrence(store)
            # Index by IDS (the on-disk group name); a recorder writes one IDS
            # per port, so the group name is the natural stream identifier.
            for group in store_mod.list_groups(store):
                index.setdefault(comp, {}).setdefault(group, {})[occ] = store
        return index

    def _discover(self) -> None:
        """Initial scan: populate the selectors and pick sensible defaults."""
        self._index = self._build_index()
        self.param.component.objects = sorted(self._index)
        if self._index and self.component not in self._index:
            self.component = sorted(self._index)[0]  # triggers the cascade
        else:
            self._on_component()

    def _refresh_index(self) -> None:
        """Poll-time rescan: surface new occurrences/stores in the dropdowns
        without disturbing the current (still-valid) selection."""
        self._index = self._build_index()
        ids_streams = self._index.get(self.component, {})
        self.param.component.objects = sorted(self._index)
        self.param.ids.objects = sorted(ids_streams)
        self.param.outer_loop.objects = sorted(ids_streams.get(self.ids, {}))

    @param.depends("component", watch=True)
    def _on_component(self) -> None:
        ids_streams = self._index.get(self.component, {})
        self.param.ids.objects = sorted(ids_streams)
        if ids_streams and self.ids not in ids_streams:
            self.ids = sorted(ids_streams)[0]
        else:
            self._on_ids()

    @param.depends("ids", watch=True)
    def _on_ids(self) -> None:
        occ = self._index.get(self.component, {}).get(self.ids, {})
        labels = sorted(occ)
        self.param.outer_loop.objects = labels
        # Default to the latest occurrence (most recent iteration).
        if labels and self.outer_loop not in occ:
            self.outer_loop = labels[-1]
        else:
            self._on_outer_loop()

    @param.depends("outer_loop", watch=True)
    def _on_outer_loop(self) -> None:
        self._reload()

    def _store(self) -> Optional[Path]:
        return (
            self._index.get(self.component, {})
            .get(self.ids, {})
            .get(self.outer_loop)
        )

    def _reload(self) -> None:
        """(Re)open the resolved store/group; refresh variable/time options."""
        store = self._store()
        if not store or not self.ids:
            self._ds = None
            self.param.variables.objects = []
            self.variables = []
            self._bump()
            return
        try:
            self._ds = store_mod.open_group(store, self.ids)
        except Exception:
            logger.warning(
                "failed to open %s/%s", store, self.ids, exc_info=True
            )
            return
        options = store_mod.plottable_variables(self._ds)
        self.param.variables.objects = options
        kept = [v for v in self.variables if v in options]
        # Preselect this IDS's default quantities (reproducing the old inline
        # visualization actor); fall back to the first variable so the grid is
        # never empty on first load.
        patterns = _DEFAULT_VARS.get(self.ids, [])
        preselected = [
            v
            for v in options
            if "_error" not in v and any(p in v for p in patterns)
        ]
        self.variables = (
            kept or preselected or (options[:1] if options else [])
        )
        self._last_n = 0
        self._stale_polls = 0
        self._writing = True
        self._update_time_bounds(follow_latest=True)
        self._bump()

    def _bump(self) -> None:
        """Force the (DynamicMap) grid to rebuild against the new dataset."""
        self._struct += 1

    def _update_time_bounds(self, follow_latest: bool = False) -> None:
        n = (
            self._ds.sizes.get(store_mod.TIME, 0)
            if self._ds is not None
            else 0
        )
        self.param.time_index.bounds = (0, max(0, n - 1))
        if follow_latest and n:
            self.time_index = n - 1

    # --- live tail + animation ---------------------------------------------

    def refresh(self) -> None:
        """Data poll: surface new occurrences and pick up appended steps."""
        self._refresh_index()
        store = self._store()
        if store is None or not self.ids:
            return
        try:
            self._ds = store_mod.open_group(store, self.ids)
        except Exception:
            return
        n = self._ds.sizes.get(store_mod.TIME, 0)
        if n > self._last_n:
            self._stale_polls = 0
            self._writing = True
        else:
            self._stale_polls += 1
            if self._stale_polls >= _FINISHED_AFTER_STALE_POLLS:
                self._writing = False
        self._last_n = n
        self._update_time_bounds()

    def advance(self) -> None:
        """Player tick: step time forward; hold at the end while still writing.

        At the last step the player only wraps to the start once the run looks
        finished (no new steps for a couple of polls); while data is still
        being appended it stays at the latest frame, tracking the live front.
        """
        if not self.playing or self._ds is None:
            return
        n = self._ds.sizes.get(store_mod.TIME, 0)
        if n <= 1:
            return
        if self.time_index < n - 1:
            self.time_index += 1
        elif not self._writing:
            self.time_index = 0

    # --- view ---------------------------------------------------------------

    def _groups(self, variables: list) -> list:
        """Group selected variables so same-units peers overlay in one plot.

        Rank-0 (over time) and rank-1 (profile) variables that share units —
        and, for profiles, the same coordinate — are grouped together;
        everything else (rank-2 maps, unitless variables) plots on its own.
        """
        groups: dict = {}
        order: list = []
        for var in variables:
            da = self._ds[var]  # type: ignore[index]
            rank = store_mod.variable_rank(da)
            units = da.attrs.get("units")
            if rank in (0, 1) and units:
                key: tuple = (rank, units, tuple(store_mod.coord_names(da)))
            else:
                key = ("solo", var)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(var)
        return [groups[key] for key in order]

    def _make_plot_fn(self, group: list) -> Callable[[int], hv.Element]:
        def _fn(time_index: int) -> hv.Element:
            ds = self._ds
            if ds is None:
                return hv.Curve(([], [])).opts(responsive=True)
            try:
                return plot_overlay(ds, group, time_index)
            except Exception:
                logger.warning("plot failed for %s", group, exc_info=True)
                return hv.Curve(([], [])).opts(
                    title="plot error", responsive=True
                )

        return _fn

    @param.depends("variables", "_struct")
    def _grid(self) -> Viewable:
        if self._ds is None:
            return pn.pane.Markdown(
                "### Waiting for data — pick a Component / IDS."
            )
        if not self.variables:
            return pn.pane.Markdown(
                "### Select one or more variables to plot."
            )
        # Over-time plots (rank 0) all share the time x-axis, so collect them
        # in one linked column; slice plots (profiles/maps, rank 1-2) — where
        # the time slider picks the slice — are independent tiles.
        time_maps = []
        slice_tiles = []
        for group in self._groups(self.variables):
            dmap = hv.DynamicMap(
                param.bind(
                    self._make_plot_fn(group),
                    time_index=self.param.time_index,
                )
            ).opts(framewise=True)
            if store_mod.variable_rank(self._ds[group[0]]) == 0:
                time_maps.append(dmap)
            else:
                slice_tiles.append(
                    pn.pane.HoloViews(
                        dmap,
                        sizing_mode="stretch_both",
                        min_height=340,
                        min_width=360,
                    )
                )
        tiles = []
        if time_maps:
            # Stack the over-time plots with a single, shared time axis: link
            # their ranges (pan/zoom together) and draw the x-axis only on the
            # bottom panel instead of repeating it on each.
            last = len(time_maps) - 1
            stacked = [
                dmap if i == last else dmap.opts(xaxis=None)
                for i, dmap in enumerate(time_maps)
            ]
            linked = hv.Layout(stacked).cols(1).opts(shared_axes=True)
            tiles.append(
                pn.pane.HoloViews(
                    linked,
                    sizing_mode="stretch_width",
                    min_height=140 * len(stacked),
                )
            )
        tiles.extend(slice_tiles)
        return pn.FlexBox(*tiles, sizing_mode="stretch_width")

    @param.depends("time_index")
    def _time_label(self) -> Viewable:
        if self._ds is None:
            return pn.pane.Markdown("")
        times = self._ds[store_mod.TIME].values  # type: ignore[index]
        if not len(times):
            return pn.pane.Markdown("")
        i = max(0, min(self.time_index, len(times) - 1))
        return pn.pane.Markdown(
            f"**t = {float(times[i]):.4g} s**  ({i + 1}/{len(times)})"
        )

    def __panel__(self) -> Viewable:
        selectors = pn.Row(
            pn.widgets.Select.from_param(
                self.param.component, name="Component"
            ),
            pn.widgets.Select.from_param(self.param.ids, name="IDS"),
            pn.widgets.Select.from_param(
                self.param.outer_loop, name="Outer loop"
            ),
        )
        variables = pn.widgets.MultiChoice.from_param(
            self.param.variables, name="Variables", sizing_mode="stretch_width"
        )
        controls = pn.Row(
            pn.widgets.Toggle.from_param(
                self.param.playing, name="▶ Animate", width=110
            ),
            pn.widgets.IntSlider.from_param(
                self.param.time_index, name="Time index"
            ),
            self._time_label,
        )
        return pn.Column(
            selectors,
            variables,
            controls,
            self._grid,
            sizing_mode="stretch_width",
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

    A profile-stamped run shows a tab per profile (its bespoke, always-present
    plots) plus a generic "IMAS plots" tab; a plain run shows just the generic
    viewer.
    """
    run_dir = Path(run_dir)
    live = pn.state.curdoc is not None

    def mount(view: object) -> object:
        if live and hasattr(view, "refresh"):
            pn.state.add_periodic_callback(view.refresh, REFRESH_MS)
        # The generic viewer animates; drive its player during the session.
        if live and hasattr(view, "advance"):
            pn.state.add_periodic_callback(view.advance, PLAY_MS)
        return view

    profiles = _profiles_for(run_dir)
    browser = mount(ImasPlotsViewer(run_dir))
    if not profiles:
        return browser

    tabs = pn.Tabs(sizing_mode="stretch_width")
    for path in profiles:
        try:
            tabs.append((Path(path).stem, mount(ProfileView(run_dir, path))))
        except Exception:
            logger.warning("could not load profile %s", path, exc_info=True)
    tabs.append(("IMAS plots", browser))
    return tabs


def make_panel(run_dir: Path) -> Optional[RunPanel]:
    """Plugin entry point: an IMAS-plots panel, or None if no stores."""
    if not store_mod.find_stores(Path(run_dir)):
        return None
    return RunPanel(title="IMAS plots", view=lambda: _mounted_view(run_dir))


def serve(run_dir: Path, port: int = 0, show: bool = True) -> None:
    """Serve the viewer standalone (without the dashboard)."""
    pn.serve(
        lambda: _mounted_view(run_dir),
        port=port,
        show=show,
        title="IMAS plots",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Browse recorded IMAS data (Zarr stores) for a run."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory to scan.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    serve(args.run_dir, port=args.port, show=not args.no_show)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

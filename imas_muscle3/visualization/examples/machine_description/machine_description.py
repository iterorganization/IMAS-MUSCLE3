"""
Example that plots the following:
- First wall and divertor from machine description IDS.
- Boundary outline from an equilibrium IDS
"""

import holoviews as hv
import numpy as np
import panel as pn
import xarray as xr

from imas_muscle3.visualization.base_plotter import BasePlotter
from imas_muscle3.visualization.base_state import BaseState


class State(BaseState):
    def extract(self, ids):
        if ids.metadata.name == "equilibrium":
            self.extract_equilibrium(ids)

    def extract_equilibrium(self, ids):
        ts = ids.time_slice[0]
        outline = ts.boundary.outline

        boundary_data = xr.Dataset(
            {
                "r": (("time", "point"), [outline.r]),
                "z": (("time", "point"), [outline.z]),
            },
            coords={
                "time": [ids.time[0]],
                "point": np.arange(len(outline.r)),
            },
        )

        current_data = self.data.get("equilibrium")
        if current_data is None:
            self.data["equilibrium"] = boundary_data
        else:
            self.data["equilibrium"] = xr.concat(
                [current_data, boundary_data], dim="time", join="outer"
            )


class Plotter(BasePlotter):
    DEFAULT_OPTS = hv.opts.Overlay(
        xlim=(0, 13),
        ylim=(-10, 10),
        title="Wall and equilibrium boundary outline",
        xlabel="r [m]",
        ylabel="z [m]",
    )

    def get_dashboard(self):
        elements = [
            hv.DynamicMap(self._plot_boundary_outline),
            hv.DynamicMap(self._plot_wall),
        ]
        overlay = hv.Overlay(elements).collate().opts(self.DEFAULT_OPTS)
        return pn.pane.HoloViews(overlay, width=800, height=1000)

    @pn.depends("time")
    def _plot_boundary_outline(self):
        state = self.active_state.data.get("equilibrium")

        if state is not None and "r" in state and "z" in state:
            selected_data = state.sel(time=self.time)
            r = selected_data.r.values
            z = selected_data.z.values
        else:
            r, z = [], [], "Waiting for data..."

        return hv.Curve((r, z)).opts(self.DEFAULT_OPTS)

    def _plot_wall(self):
        """Generates path for limiter and divertor."""
        paths = []
        wall = self.active_state.md.get("wall")
        if wall is not None:
            for unit in wall.description_2d[0].limiter.unit:
                name = str(unit.name)
                r_vals = unit.outline.r
                z_vals = unit.outline.z
                paths.append((r_vals, z_vals, name))
        return hv.Path(paths, vdims=["name"]).opts(
            color="black",
            line_width=2,
            hover_tooltips=[("", "@name")],
        )

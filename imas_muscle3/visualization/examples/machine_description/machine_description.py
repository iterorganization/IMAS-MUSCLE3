import logging

import holoviews as hv
import panel as pn
import xarray as xr

from imas_muscle3.visualization.base_plotter import BasePlotter
from imas_muscle3.visualization.base_state import BaseState

logger = logging.getLogger()


class State(BaseState):
    def extract(self, ids):
        if ids.metadata.name == "equilibrium":
            self._extract_equilibrium(ids)

    def _extract_equilibrium(self, ids):
        ts = ids.time_slice[0]
        prof2d = ts.profiles_2d[0]
        psi = prof2d.psi
        new_point = xr.Dataset(
            {
                "psi": (("dim1", "dim2"), psi.value),
            },
            coords={
                "time": [ids.time[0]],
                "dim1": prof2d.grid.dim1,
                "dim2": prof2d.grid.dim2,
            },
        )

        current_data = self.data.get("equilibrium")
        if current_data is None:
            self.data["equilibrium"] = new_point
        else:
            self.data["equilibrium"] = xr.concat(
                [current_data, new_point.expand_dims("time")], dim="time", join="outer"
            )


class Plotter(BasePlotter):
    DEFAULT_OPTS = hv.opts.Overlay(
        xlim=(0, 13),
        ylim=(-10, 10),
        title="Equilibrium poloidal flux",
        xlabel="r [m]",
        ylabel="z [m]",
    )
    QUADMESH_OPTS = hv.opts.QuadMesh(
        cmap="viridis",
        colorbar=True,
        colorbar_opts={"title": "Poloidal flux [Wb]"},
        tools=["hover"],
        aspect="equal",
        framewise=True,
    )

    def get_dashboard(self):
        flux_map_elements = [
            hv.DynamicMap(self._plot_psi_quadmesh),
            hv.DynamicMap(self._plot_wall),
        ]
        flux_map_overlay = (
            hv.Overlay(flux_map_elements).collate().opts(self.DEFAULT_OPTS)
        )
        return pn.pane.HoloViews(flux_map_overlay, width=800, height=1000)

    @pn.depends("time")
    def _plot_psi_quadmesh(self):
        """Plots 2D poloidal flux as QuadMesh."""
        state = self.active_state.data.get("equilibrium")
        if state is None:
            return hv.QuadMesh(([0, 1], [0, 1], np.zeros((1, 1))))
        selected_data = state.sel(time=self.time)
        r = selected_data.grid_r.values
        z = selected_data.grid_z.values
        psi = selected_data.psi.values
        return hv.QuadMesh((r, z, psi)).opts(self.QUADMESH_OPTS)

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

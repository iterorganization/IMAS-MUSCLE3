"""
Example that plots the following:
- a contour plot of the poloidal flux, along with the separatrix,
  and X/O-points from an equilibrium IDS.
- The inner wall / vacuum vessel from a wall machine description IDS.
- The outline of the coils from a pf_active machine description IDS.
- 1D profile and waterfall plot of ff' profile.
- 1D profile and waterfall plot of p' profile.
- Coil currents over time.
"""

import logging

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import xarray as xr

from imas_muscle3.visualization.base import BasePlotter, BaseState

logger = logging.getLogger()


class State(BaseState):
    def extract(self, ids):
        if ids.metadata.name == "equilibrium":
            self._extract_equilibrium(ids)
        elif ids.metadata.name == "pf_active":
            self._extract_pf_active(ids)

    def _extract_pf_active(self, ids):
        currents = np.array([c.current.data for c in ids.coil])
        coil_names = np.array([c.name.value for c in ids.coil])
        ncoils = len(ids.coil)
        new_point = xr.Dataset(
            {
                "currents": (("time", "coil"), currents.reshape(1, ncoils)),
            },
            coords={"time": [ids.time[0]], "coil": coil_names},
        )

        current_data = self.data.get("pf_active")
        if current_data is None:
            self.data["pf_active"] = new_point
        else:
            self.data["pf_active"] = xr.concat(
                [current_data, new_point], dim="time", join="outer"
            )

    def _extract_equilibrium(self, ids):
        ts = ids.time_slice[0]

        # Extract separatrix data
        separatrix_data = xr.Dataset(
            {
                "r": (("time", "point"), [ts.boundary.outline.r]),
                "z": (("time", "point"), [ts.boundary.outline.z]),
            },
            coords={
                "time": [ids.time[0]],
                "point": range(len(ts.boundary.outline.r)),
            },
        )

        # Extract grid data for contours
        eqggd = ts.ggd[0]
        r_vals = eqggd.r[0].values
        z_vals = eqggd.z[0].values
        psi_vals = eqggd.psi[0].values

        grid_data = xr.Dataset(
            {
                "grid_r": (("time", "grid_point"), [r_vals]),
                "grid_z": (("time", "grid_point"), [z_vals]),
                "psi": (("time", "grid_point"), [psi_vals]),
                "boundary_psi": (("time",), [ts.boundary.psi]),
            },
            coords={
                "time": [ids.time[0]],
                "grid_point": range(len(r_vals)),
            },
        )

        # Extract X-point and O-point data
        x_points_r = []
        x_points_z = []
        o_points_r = []
        o_points_z = []

        for node in ts.contour_tree.node:
            if node.critical_type == 1:  # X-point
                x_points_r.append(node.r)
                x_points_z.append(node.z)
            elif node.critical_type == 0 or node.critical_type == 2:  # O-point
                o_points_r.append(node.r)
                o_points_z.append(node.z)

        critical_points_data = xr.Dataset(
            {
                "x_points_r": (("time", "x_point"), [x_points_r]),
                "x_points_z": (("time", "x_point"), [x_points_z]),
                "o_points_r": (("time", "o_point"), [o_points_r]),
                "o_points_z": (("time", "o_point"), [o_points_z]),
            },
            coords={
                "time": [ids.time[0]],
                "x_point": range(len(x_points_r)),
                "o_point": range(len(o_points_r)),
            },
        )

        # Extract profiles
        profiles_data = xr.Dataset(
            {
                "f_df_dpsi": (("time", "profile"), [ts.profiles_1d.f_df_dpsi]),
                "dpressure_dpsi": (
                    ("time", "profile"),
                    [ts.profiles_1d.dpressure_dpsi],
                ),
                "psi_profile": (("time", "profile"), [ts.profiles_1d.psi]),
            },
            coords={
                "time": [ids.time[0]],
                "profile": np.arange(len(ts.profiles_1d.f_df_dpsi)),
            },
        )

        # Combine all datasets
        new_data = xr.merge(
            [separatrix_data, grid_data, critical_points_data, profiles_data]
        )

        current_data = self.data.get("equilibrium")
        if current_data is None:
            self.data["equilibrium"] = new_data
        else:
            self.data["equilibrium"] = xr.concat(
                [current_data, new_data], dim="time", join="outer"
            )


class Plotter(BasePlotter):
    WIDTH = 800
    HEIGHT = 1000

    PROFILE_WIDTH = 350
    PROFILE_HEIGHT = 350
    DEFAULT_OPTS = hv.opts.Overlay(
        xlim=(0, 13),
        ylim=(-10, 10),
        title="Equilibrium poloidal flux",
        xlabel="r [m]",
        ylabel="z [m]",
    )
    CONTOUR_OPTS = hv.opts.Contours(
        cmap="viridis",
        colorbar=True,
        tools=["hover"],
        colorbar_opts={"title": "Poloidal flux [Wb]"},
        show_legend=False,
    )
    DESIRED_SHAPE_OPTS = hv.opts.Curve(color="blue")

    levels = param.Integer(default=20, bounds=(1, 100), doc="Number of contour levels")

    def get_dashboard(self):
        # Create poloidal flux plot
        flux_map_elements = [
            hv.DynamicMap(self._plot_contours),
            hv.DynamicMap(self._plot_separatrix),
            hv.DynamicMap(self._plot_xo_points),
            hv.DynamicMap(self._plot_coil_rectangles),
            hv.DynamicMap(self._plot_wall),
            hv.DynamicMap(self._plot_vacuum_vessel),
        ]
        contour_slider = pn.widgets.IntSlider.from_param(
            self.param.levels, name="Contour levels"
        )
        flux_map_overlay = (
            hv.Overlay(flux_map_elements).collate().opts(self.DEFAULT_OPTS)
        )

        coil_currents = hv.DynamicMap(self.plot_coil_currents)
        f_df_dpsi = hv.DynamicMap(self.plot_f_df_dpsi_profile)
        f_df_dpsi_2d = hv.DynamicMap(self.plot_f_df_dpsi_2d)
        dpressure_dpsi = hv.DynamicMap(self.plot_dpressure_dpsi)
        dpressure_dpsi_2d = hv.DynamicMap(self.plot_dpressure_dpsi_2d)

        return pn.Row(
            pn.Column(
                contour_slider,
                pn.pane.HoloViews(
                    flux_map_overlay, width=self.WIDTH, height=self.HEIGHT
                ),
            ),
            pn.Column(
                pn.Row(f_df_dpsi, f_df_dpsi_2d),
                pn.Row(dpressure_dpsi, dpressure_dpsi_2d),
                coil_currents,
            ),
        )

    def _plot_coil_rectangles(self):
        """Creates rectangular and path overlays for PF coils.

        Returns:
            Coil geometry overlay.
        """
        pf_active = self.active_state.md["pf_active"]
        rectangles = []
        paths = []
        if pf_active is not None:
            for idx, coil in enumerate(pf_active.coil):
                name = str(coil.name)
                for element in coil.element:
                    rect = element.geometry.rectangle
                    outline = element.geometry.outline
                    annulus = element.geometry.annulus
                    if rect.has_value:
                        r0 = rect.r - rect.width / 2
                        r1 = rect.r + rect.width / 2
                        z0 = rect.z - rect.height / 2
                        z1 = rect.z + rect.height / 2
                        rectangles.append((r0, z0, r1, z1, name))
                    elif outline.has_value:
                        paths.append((outline.r, outline.z, name))
                    elif annulus.r.has_value:
                        phi = np.linspace(0, 2 * np.pi, 17)
                        paths.append(
                            (
                                (annulus.r + annulus.radius_outer * np.cos(phi)),
                                (annulus.z + annulus.radius_outer * np.sin(phi)),
                                name,
                            )
                        )
                    else:
                        logger.warning(
                            f"Coil {name} was skipped, as it does not have a filled "
                            "'rect' or 'outline' node"
                        )
                        continue
        rects = hv.Rectangles(rectangles, vdims=["name"]).opts(
            line_color="black",
            fill_alpha=0,
            line_width=2,
            show_legend=False,
            hover_tooltips=[("", "@name")],
        )
        paths = hv.Path(paths, vdims=["name"]).opts(
            color="black",
            line_width=1,
            show_legend=False,
            hover_tooltips=[("", "@name")],
        )
        return rects * paths

    @pn.depends("time_index", "levels")
    def _plot_contours(self):
        """Generates contour plot for poloidal flux.

        Returns:
            Contour plot of psi.
        """
        state = self.active_state.data.get("equilibrium")
        if state is None:
            contours = hv.Contours(([0], [0], 0), vdims="psi")
        else:
            selected_data = state.isel(time=self.time_index)
            contours = self._calc_contours(selected_data, self.levels)
        return contours.opts(self.CONTOUR_OPTS)

    def _calc_contours(self, equilibrium_data, levels):
        """Calculates the contours of the psi grid of an equilibrium dataset.

        Args:
            equilibrium_data: The equilibrium dataset to load psi grid from.
            levels: Determines the number of contour lines. Either an integer for total
                number of contour lines, or a list of specified levels.

        Returns:
            Holoviews contours object
        """
        r = equilibrium_data.grid_r.values
        z = equilibrium_data.grid_z.values
        psi = equilibrium_data.psi.values

        trics = plt.tricontour(r, z, psi, levels=levels)
        return hv.Contours(self._extract_contour_segments(trics), vdims="psi")

    def _extract_contour_segments(self, tricontour):
        """Extracts contour segments from matplotlib tricontour.

        Args:
            tricontour: Output from plt.tricontour.

        Returns:
            Segment dictionaries with 'x', 'y', and 'psi'.
        """
        segments = []
        for i, level in enumerate(tricontour.levels):
            for seg in tricontour.allsegs[i]:
                if len(seg) > 1:
                    segments.append({"x": seg[:, 0], "y": seg[:, 1], "psi": level})
        return segments

    @pn.depends("time_index")
    def _plot_separatrix(self):
        """Plots the separatrix from the equilibrium boundary.

        Returns:
            Holoviews curve containing the separatrix.
        """

        state = self.active_state.data.get("equilibrium")
        if state is None:
            r = z = []
            contour = hv.Contours(([0], [0], 0), vdims="psi")
        else:
            selected_data = state.isel(time=self.time_index)
            r = selected_data.r
            z = selected_data.z

            # Get boundary psi and create contour at that level
            boundary_psi = selected_data.boundary_psi.values
            contour = self._calc_contours(selected_data, [boundary_psi])
        return hv.Curve((r, z)).opts(
            color="red",
            line_width=4,
            show_legend=False,
            hover_tooltips=[("", "Separatrix")],
        ) * contour.opts(self.CONTOUR_OPTS)

    def _plot_vacuum_vessel(self):
        """Generates path for inner and outer vacuum vessel.

        Returns:
            Holoviews path containing the geometry.
        """
        paths = []
        wall = self.active_state.md["wall"]
        if wall is not None:
            for unit in wall.description_2d[0].vessel.unit:
                name = str(unit.name)
                r_vals = unit.annular.centreline.r
                z_vals = unit.annular.centreline.z
                paths.append((r_vals, z_vals, name))
        return hv.Path(paths, vdims=["name"]).opts(
            color="black",
            line_width=2,
            hover_tooltips=[("", "@name")],
        )

    def _plot_wall(self):
        """Generates path for limiter and divertor.

        Returns:
            Holoviews path containing the geometry.
        """
        paths = []
        wall = self.active_state.md["wall"]
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

    @pn.depends("time_index")
    def _plot_xo_points(self):
        """Plots X-points and O-points from the equilibrium.

        Returns:
            Scatter plots of X and O points.
        """
        o_points = []
        x_points = []

        equilibrium = self.active_state.data.get("equilibrium")
        if equilibrium is not None:
            selected_data = equilibrium.isel(time=self.time_index)

            # Extract X-points
            x_r = selected_data.x_points_r.values
            x_z = selected_data.x_points_z.values
            x_points = list(zip(x_r, x_z))

            # Extract O-points
            o_r = selected_data.o_points_r.values
            o_z = selected_data.o_points_z.values
            o_points = list(zip(o_r, o_z))

        o_scatter = hv.Scatter(o_points).opts(
            marker="o",
            size=10,
            color="black",
            show_legend=False,
            hover_tooltips=[("", "O-point")],
        )
        x_scatter = hv.Scatter(x_points).opts(
            marker="x",
            size=10,
            color="black",
            show_legend=False,
            hover_tooltips=[("", "X-point")],
        )
        return o_scatter * x_scatter

    @param.depends("time_index")
    def plot_f_df_dpsi_profile(self):
        xlabel = "Psi"
        ylabel = "ff'"
        state = self.active_state.data.get("equilibrium")

        if state:
            selected_data = state.isel(time=self.time_index)
            psi = selected_data.psi_profile
            f_df_dpsi = selected_data.f_df_dpsi
            title = "ff' profile"
        else:
            psi, f_df_dpsi, title = [], [], "Waiting for data..."

        return hv.Curve((psi, f_df_dpsi), xlabel, ylabel).opts(
            framewise=True, height=300, width=600, title=title
        )

    @param.depends("time_index")
    def plot_dpressure_dpsi(self):
        xlabel = "Psi"
        ylabel = "p'"
        state = self.active_state.data.get("equilibrium")

        if state:
            selected_data = state.isel(time=self.time_index)
            psi = selected_data.psi_profile
            dpressure_dpsi = selected_data.dpressure_dpsi
            title = "p' profile"
        else:
            psi, dpressure_dpsi, title = [], [], "Waiting for data..."

        return hv.Curve((psi, dpressure_dpsi), xlabel, ylabel).opts(
            framewise=True, height=300, width=600, title=title
        )

    @param.depends("time_index")
    def plot_coil_currents(self):
        state = self.active_state.data.get("pf_active")
        xlabel = "Time [s]"
        ylabel = "Coil currents [A]"

        if state:
            time = state.time[: self.time_index + 1]
            curves = []
            for coil in state.coil.values:
                current = state.currents.sel(coil=coil)[: self.time_index + 1].values
                curve = hv.Curve((time, current), xlabel, ylabel, label=str(coil)).opts(
                    framewise=True,
                    title="coil currents over time",
                )
                curves.append(curve)
        else:
            curves = [hv.Curve(([0], [0]), xlabel, ylabel)]

        return hv.Overlay(curves).opts(height=450, width=1200)

    @param.depends("time_index")
    def plot_f_df_dpsi_2d(self):
        state = self.active_state.data.get("equilibrium")
        ylabel = "Time [s]"
        xlabel = "psi"

        if state:
            times = state.time.values[: self.time_index + 1]
            psi = state.psi_profile.values[self.time_index]
            f_df_dpsi = state.f_df_dpsi.values[: self.time_index + 1, :]
            title = "ff' over time"
        else:
            times = np.array([0, 1])
            psi = np.array([0, 1])
            f_df_dpsi = np.full((2, 2), 0)
            title = "Waiting for data..."

        return hv.QuadMesh(
            (psi, times, f_df_dpsi),
            kdims=["profile_dim", "time_dim"],
            vdims=["ffprime"],
        ).opts(
            cmap="viridis",
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=True,
            framewise=True,
            height=300,
            width=600,
            title=title,
        )

    @param.depends("time_index")
    def plot_dpressure_dpsi_2d(self):
        state = self.active_state.data.get("equilibrium")
        ylabel = "Time [s]"
        xlabel = "psi"

        if state:
            times = state.time.values[: self.time_index + 1]
            psi = state.psi_profile.values[self.time_index]
            dpressure_dpsi = state.dpressure_dpsi.values[: self.time_index + 1, :]
            title = "p' over time"
        else:
            times = np.array([0, 1])
            psi = np.array([0, 1])
            dpressure_dpsi = np.full((2, 2), 0)
            title = "Waiting for data..."

        return hv.QuadMesh(
            (psi, times, dpressure_dpsi),
            kdims=["profile_dim", "time_dim"],
            vdims=["ffprime"],
        ).opts(
            cmap="viridis",
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=True,
            framewise=True,
            height=300,
            width=600,
            title=title,
        )

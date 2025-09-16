import holoviews as hv
import numpy as np
import panel as pn
import param
import xarray as xr

from imas_muscle3.visualization.base_state import BasePlotter, BaseState


class State(BaseState):
    def extract(self, ids):
        if ids.metadata.name == "equilibrium":
            self._extract_equilibrium(ids)
        elif ids.metadata.name == "pf_active":
            self._extract_pf_active(ids)

    def _extract_equilibrium(self, ids):
        ts = ids.time_slice[0]
        new_point = xr.Dataset(
            {
                "ip": ("time", [ts.global_quantities.ip]),
                "f_df_dpsi": (("time", "profile"), [ts.profiles_1d.f_df_dpsi]),
                "psi": (("time", "profile"), [ts.profiles_1d.psi]),
                "profiles_2d": (
                    ("time", "r", "z"),
                    [ts.profiles_2d[0].psi],
                ),
            },
            coords={
                "time": [ids.time[0]],
                "profile": np.arange(len(ts.profiles_1d.f_df_dpsi)),
                "r": ts.profiles_2d[0].grid.dim1,
                "z": ts.profiles_2d[0].grid.dim2,
            },
        )

        current_data = self.data.get("equilibrium")
        if current_data is None:
            self.data["equilibrium"] = new_point
        else:
            self.data["equilibrium"] = xr.concat(
                [current_data, new_point], dim="time", join="outer"
            )

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


class Plotter(BasePlotter):
    def get_plots(self):
        ip_vs_time = hv.DynamicMap(self.plot_ip_vs_time)
        f_profile = hv.DynamicMap(self.plot_f_df_dpsi_profile)
        profile_2d = hv.DynamicMap(self.plot_2d_profile)
        return pn.GridBox(ip_vs_time, f_profile, profile_2d, ncols=2)

    @param.depends("time_slider", "live_view")
    def plot_ip_vs_time(self):
        xlabel = "Time [s]"
        ylabel = "Ip [A]"
        state_data = self.active_state.data.get("equilibrium")

        if state_data:
            time = state_data.time[0 : self.time_idx]
            ip = state_data.ip[0 : self.time_idx]
            current_time = state_data.time[self.time_idx].item()
            title = f"Ip over time, showing t={current_time:.3f} ({len(time)} points)"
        else:
            time, ip, title = [], [], "Waiting for data..."

        return hv.Curve((time, ip), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
        )

    @param.depends("time_slider", "live_view")
    def plot_f_df_dpsi_profile(self):
        xlabel = "Psi"
        ylabel = "ff'"
        state_data = self.active_state.data.get("equilibrium")

        if state_data:
            selected_data = state_data.isel(time=self.time_idx)
            psi = selected_data.psi
            f_df_dpsi = selected_data.f_df_dpsi
            title = f"ff' profile at t={selected_data.time.item():.3f}"
        else:
            psi, f_df_dpsi, title = [], [], "Waiting for data..."

        return hv.Curve((psi, f_df_dpsi), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
        )

    @param.depends("time_slider", "live_view")
    def plot_2d_profile(self):
        xlabel = "r"
        ylabel = "z"
        state_data = self.active_state.data.get("equilibrium")

        if state_data:
            selected_data = state_data.isel(time=self.time_idx)
            f_2d = selected_data.profiles_2d.values
            r = selected_data.coords[xlabel]
            z = selected_data.coords[ylabel]
            title = f"Poloidal flux at t={selected_data.time.item():.3f}"
        else:
            r, z, f_2d, title = [], [], [], "Waiting for data..."

        return hv.QuadMesh((r, z, f_2d.T)).opts(
            cmap="viridis",
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=True,
            framewise=True,
            height=300,
            width=960,
            title=title,
        )

    # # TODO: this plot sometimes doesn't update properly
    # @param.depends("state.data")
    # def plot_profile_waterfall(self):
    #     state = self.state.data.get("equilibrium")
    #     if state:
    #         times = state.time.values
    #         profiles = state.profile.values
    #         f_values_2d = state.f_df_dpsi.values
    #     else:
    #         times = []
    #         profiles = []
    #         f_values_2d = []
    #
    #     df = pd.DataFrame(
    #         {
    #             "Time": np.repeat(times, len(profiles)),
    #             "Psi Index": np.tile(profiles, len(times)),
    #             "ff'": f_values_2d.flatten(),
    #         }
    #     )
    #
    #     return hv.HeatMap(df, kdims=["Psi Index", "Time"], vdims=["ff'"]).opts(
    #         cmap="viridis",
    #         colorbar=True,
    #         framewise=True,
    #         height=300,
    #         width=960,
    #         title="ff' over time",
    #     )

    # @param.depends("state.data")
    # def plot_coil_currents(self):
    #     state = self.state.data.get("pf_active")
    #     xlabel = "Time [s]"
    #     ylabel = "Coil currents [A]"
    #
    #     if state:
    #         curves = [
    #             hv.Curve(
    #                 (state.time, state.currents.sel(coil=coil)),
    #                 xlabel,
    #                 ylabel,
    #                 label=str(coil),
    #             ).opts(
    #                 framewise=True,
    #                 height=500,
    #                 width=960,
    #                 title=f"coil currents over time, current t={state.time[-1].item():.3f}",
    #             )
    #             for coil in state.coil.values
    #         ]
    #     else:
    #         curves = [hv.Curve(([0, 1, 2], [0, 1, 2]), xlabel, ylabel)]
    #
    #     return hv.Overlay(curves)

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr

from imas_muscle3.visualization.base_state import BasePlotter, BaseState


class State(BaseState):
    def extract(self, ids):
        if not ids:
            return
        if ids.metadata.name == "equilibrium":
            self._extract_equilibrium(ids)
        elif ids.metadata.name == "pf_active":
            self._extract_pf_active(ids)
        self.param.trigger("data")

    def _extract_equilibrium(self, ids):
        ts = ids.time_slice[0]
        new_point = xr.Dataset(
            {
                "ip": ("time", [ts.global_quantities.ip]),
                "f_df_dpsi": (("time", "profile"), [ts.profiles_1d.f_df_dpsi]),
                "psi": (("time", "profile"), [ts.profiles_1d.psi]),
                "profiles_2d": (
                    ("time", "profile_dim1", "profile_dim2"),
                    [ts.profiles_2d[0].psi],
                ),
            },
            coords={
                "time": [ids.time[0]],
                "profile": np.arange(len(ts.profiles_1d.f_df_dpsi)),
                "profile_dim1": np.arange(ts.profiles_2d[0].psi.shape[0]),
                "profile_dim2": np.arange(ts.profiles_2d[0].psi.shape[1]),
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
    @param.depends("state.data")
    def plot_ip_vs_time(self):
        xlabel = "Time [s]"
        ylabel = "Ip [A]"
        state = self.state.data.get("equilibrium")

        if state:
            time = state.time
            ip = state.ip
            title = (
                f"Ip over time, current t={state.time[-1].item():.3f} "
                f"len={len(state.time)}"
            )
        else:
            time = []
            ip = []
            title = "Waiting for data..."

        return hv.Curve((time, ip), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
        )

    @param.depends("state.data")
    def plot_f_df_dpsi_profile(self):
        xlabel = "Psi"
        ylabel = "ff'"
        state = self.state.data.get("equilibrium")

        if state:
            latest_data = state.isel(time=-1)
            psi = latest_data.psi
            f_df_dpsi = latest_data.f_df_dpsi
            title = f"ff' profile at t={latest_data.time.item():.3f}"
        else:
            psi = []
            f_df_dpsi = []
            title = "Waiting for data..."
        return hv.Curve((psi, f_df_dpsi), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
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

    @param.depends("state.data")
    def plot_2d_profile(self):
        state = self.state.data.get("equilibrium")
        if state:
            latest_data = state.isel(time=-1)
            f_2d = latest_data.profiles_2d.values
            title = f"Poloidal flux at t={latest_data.time.item():.3f}"
        else:
            f_2d = []
            title = "Waiting for data..."

        return hv.Image(f_2d).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            height=300,
            width=960,
            title=title,
        )

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

    def get_dashboard(self):
        ip_vs_time = hv.DynamicMap(self.plot_ip_vs_time)
        f_profile = hv.DynamicMap(self.plot_f_df_dpsi_profile)
        profile_2d = hv.DynamicMap(self.plot_2d_profile)

        return pn.GridBox(ip_vs_time, f_profile, profile_2d, ncols=2)

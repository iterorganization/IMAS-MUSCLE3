import holoviews as hv
import numpy as np
import pandas as pd
import param
import xarray as xr

from imas_muscle3.visualization.base_state import BasePlotter, BaseState


class EquilibriumState(BaseState):
    def update(self, ids):
        if not ids:
            return

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

        if not self.data:
            self.data = new_point
        else:
            self.data = xr.concat([self.data, new_point], dim="time", join="outer")


class PfActiveState(BaseState):
    def update(self, ids):
        if not ids:
            return

        currents = np.array([c.current.data for c in ids.coil])
        coil_names = np.array([c.name.value for c in ids.coil])
        ncoils = len(ids.coil)

        new_point = xr.Dataset(
            {
                "currents": (("time", "coil"), currents.reshape(1, ncoils)),
            },
            coords={
                "time": [ids.time[0]],
                "coil": coil_names,
            },
        )

        if not self.data:
            self.data = new_point
        else:
            self.data = xr.concat([self.data, new_point], dim="time", join="outer")


class Plots(BasePlotter):
    @param.depends("state.data")
    def plot_ip_vs_time(self):
        xlabel = "Time [s]"
        ylabel = "Ip [A]"

        if self.state.data:
            time = self.state.data.time
            ip = self.state.data.ip
            title = (
                f"Ip over time, current t={self.state.data.time[-1].item():.3f} "
                f"len={len(self.state.data.time)}"
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

        if self.state.data:
            latest_data = self.state.data.isel(time=-1)
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

    # TODO: this plot sometimes doesn't update properly
    @param.depends("state.data")
    def plot_profile_waterfall(self):
        if self.state.data:
            times = self.state.data.time.values
            profiles = self.state.data.profile.values
            f_values_2d = self.state.data.f_df_dpsi.values
        else:
            times = []
            profiles = []
            f_values_2d = []

        df = pd.DataFrame(
            {
                "Time": np.repeat(times, len(profiles)),
                "Psi Index": np.tile(profiles, len(times)),
                "ff'": f_values_2d.flatten(),
            }
        )

        return hv.HeatMap(df, kdims=["Psi Index", "Time"], vdims=["ff'"]).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            height=300,
            width=960,
            title="ff' over time",
        )

    @param.depends("state.data")
    def plot_2d_profile(self):
        if self.state.data:
            latest_data = self.state.data.isel(time=-1)
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

    @param.depends("state.data")
    def plot_coil_currents(self):
        xlabel = "Time [s]"
        ylabel = "Coil currents [A]"

        if self.state.data:
            curves = [
                hv.Curve(
                    (self.state.data.time, self.state.data.currents.sel(coil=coil)),
                    xlabel,
                    ylabel,
                    label=str(coil),
                ).opts(
                    framewise=True,
                    height=500,
                    width=960,
                    title=f"coil currents over time, current t={self.state.data.time[-1].item():.3f}",
                )
                for coil in self.state.data.coil.values
            ]
        else:
            curves = [hv.Curve(([0, 1, 2], [0, 1, 2]), xlabel, ylabel)]

        return hv.Overlay(curves)


STATE_DEFINITIONS = {
    "equilibrium": EquilibriumState,
    "pf_active": PfActiveState,
}
DASHBOARD_LAYOUT = [
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_f_df_dpsi_profile",
    },
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_ip_vs_time",
    },
    # {
    #     "plot_class": Plots,
    #     "state_name": "equilibrium",
    #     "plot_method": "plot_profile_waterfall",
    # },
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_2d_profile",
    },
    # {
    #     "plot_class": Plots,
    #     "state_name": "pf_active",
    #     "plot_method": "plot_coil_currents",
    # },
]

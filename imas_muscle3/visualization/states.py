import holoviews as hv
import numpy as np
import pandas as pd
import param
import xarray as xr

# Import the base class from our library file
from imas_muscle3.visualization.base_state import BaseState


class EquilibriumState(BaseState):
    """Manages and processes data from the 'equilibrium' IDS."""

    def _initialize_data(self):
        self.data = xr.Dataset(
            data_vars={
                "ip": ("time", np.array([], dtype=float)),
                "f_df_dpsi": ("time", np.array([], dtype=float)),
                "psi": ("time", np.array([], dtype=float)),
            },
            coords={
                "time": np.array([], dtype=float),
            },
        )

    def update(self, equilibrium_ids):
        if not equilibrium_ids.time_slice:
            return

        ts = equilibrium_ids.time_slice[0]

        new_point = xr.Dataset(
            {
                "ip": ("time", [ts.global_quantities.ip]),
                "f_df_dpsi": (("time", "profile"), [ts.profiles_1d.f_df_dpsi]),
                "psi": (("time", "profile"), [ts.profiles_1d.psi]),
            },
            coords={
                "time": [equilibrium_ids.time[0]],
                "profile": np.arange(len(ts.profiles_1d.f_df_dpsi)),
            },
        )

        if self.data.sizes["time"] == 0:
            self.data = new_point
        else:
            self.data = xr.concat([self.data, new_point], dim="time", join="outer")


class Plots(param.Parameterized):
    """A container for plotting methods that depend on state data."""

    state = param.Parameter()

    def __init__(self, state, **params):
        super().__init__(state=state, **params)

    @param.depends("state.data")
    def plot_ip_vs_time(self):
        xlabel = "Time (s)"
        ylabel = "Ip (A)"

        if not self.state or self.state.data.time.size == 0:
            return hv.Curve(([], []), xlabel, ylabel).opts(
                framewise=True, responsive=True, height=300, title="Waiting for data..."
            )

        return hv.Curve(
            (self.state.data.time, self.state.data.ip), xlabel, ylabel
        ).opts(
            framewise=True,
            responsive=True,
            height=300,
            title=f"Ip over time, current t={self.state.data.time[-1].item()}, len={len(self.state.data.time)}",
        )

    @param.depends("state.data")
    def plot_profile(self):
        xlabel = "Psi"
        ylabel = "ff'"

        if not self.state or self.state.data.time.size == 0:
            return hv.Curve(([], []), xlabel, ylabel).opts(
                framewise=True, responsive=True, height=300, title="Waiting for data..."
            )
        latest_data = self.state.data.isel(time=-1)
        return hv.Curve((latest_data.psi, latest_data.f_df_dpsi), xlabel, ylabel).opts(
            framewise=True,
            responsive=True,
            height=300,
            title=f"Profile at t={latest_data.time.item():.6f}",
        )

    @param.depends("state.data")
    def plot_profile_waterfall(self):
        if not self.state or self.state.data.time.size == 0:
            return hv.HeatMap(pd.DataFrame(columns=["Time", "Psi Index", "ff'"])).opts(
                cmap="viridis",
                colorbar=True,
                framewise=True,
                responsive=True,
                height=400,
                title="Waterfall of ff' over Time and Psi Index",
            )

        times = self.state.data.time.values
        profiles = self.state.data.profile.values
        f_values_2d = self.state.data.f_df_dpsi.values

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
            responsive=True,
            height=400,
            title="Waterfall of ff' over Time and Psi Index",
        )


STATE_DEFINITIONS = {
    "equilibrium": EquilibriumState,
}
DASHBOARD_LAYOUT = [
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_ip_vs_time",
    },
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_profile",
    },
    {
        "plot_class": Plots,
        "state_name": "equilibrium",
        "plot_method": "plot_profile_waterfall",
    },
]

"""
Simple example plot which plots the plasma current over time.
"""

import holoviews as hv
import param
import xarray as xr

from imas_muscle3.visualization.base_plotter import BasePlotter
from imas_muscle3.visualization.base_state import BaseState


class State(BaseState):
    def extract(self, ids):
        if ids.metadata.name == "equilibrium":
            self._extract_equilibrium(ids)

    def _extract_equilibrium(self, ids):
        ts = ids.time_slice[0]
        new_point = xr.Dataset(
            {
                "ip": ("time", [ts.global_quantities.ip]),
            },
            coords={
                "time": [ids.time[0]],
            },
        )

        current_data = self.data.get("equilibrium")
        if current_data is None:
            self.data["equilibrium"] = new_point
        else:
            self.data["equilibrium"] = xr.concat(
                [current_data, new_point], dim="time", join="outer"
            )


class Plotter(BasePlotter):
    def get_dashboard(self):
        ip_vs_time = hv.DynamicMap(self.plot_ip_vs_time)
        return ip_vs_time

    @param.depends("time")
    def plot_ip_vs_time(self):
        xlabel = "Time [s]"
        ylabel = "Ip [A]"
        state = self.active_state.data.get("equilibrium")

        if state:
            mask = state.time <= self.time
            time = state.time[mask]
            ip = state.ip[mask]
            title = "Ip over time"
        else:
            time, ip, title = [], [], "Waiting for data..."

        return hv.Curve((time, ip), kdims=["time_ip"], vdims=["ip"]).opts(
            framewise=True,
            height=300,
            width=960,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

import holoviews as hv
import numpy as np
import param
import xarray as xr

# Import the base class from our library file
from imas_muscle3.visualization.base_state import BaseState


class EquilibriumState(BaseState):
    """Manages and processes data from the 'equilibrium' IDS."""

    def _initialize_data(self):
        self.data = xr.Dataset(
            data_vars={"ip": ("time", np.array([], dtype=float))},
            coords={"time": np.array([], dtype=float)},
        )

    def update(self, equilibrium_ids):
        if not equilibrium_ids.time_slice:
            return

        new_point = xr.Dataset(
            {"ip": ("time", [equilibrium_ids.time_slice[0].global_quantities.ip])},
            coords={"time": equilibrium_ids.time},
        )

        if self.data.dims["time"] == 0:
            self.data = new_point
        else:
            self.data = xr.concat([self.data, new_point], dim="time")
        self.param.trigger("data")


class Plots(param.Parameterized):
    """A container for plotting methods that depend on state data."""

    state = param.Parameter()

    def __init__(self, state, **params):
        super().__init__(state=state, **params)

    @param.depends("state.data", watch=True)
    def plot_ip_vs_time(self):
        """Plots the history of Ip vs time."""
        if not self.state or self.state.data.time.size == 0:
            return hv.Curve(([], []), "Time (s)", "Ip (A)").opts(
                framewise=True, responsive=True, height=300, title="Waiting for data..."
            )

        return hv.Curve(
            (self.state.data.time, self.state.data.ip), "Time (s)", "Ip (A)"
        ).opts(
            framewise=True,
            responsive=True,
            height=300,
            title=f"Ip over time, current t={self.state.data.time[-1].item()}, len={len(self.state.data.time)}",
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
]

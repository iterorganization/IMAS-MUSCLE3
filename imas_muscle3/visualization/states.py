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
        ff_profile = hv.DynamicMap(self.plot_f_df_dpsi_profile)
        profile_2d = hv.DynamicMap(self.plot_2d_profile)
        ff_profile_2d = hv.DynamicMap(self.plot_ffprime_profile_2d)
        coil_currents = hv.DynamicMap(self.plot_coil_currents)
        return pn.Column(
            pn.GridBox(ip_vs_time, ff_profile, profile_2d, ff_profile_2d, ncols=2),
            coil_currents,
        )

    @param.depends("time_idx")
    def plot_ip_vs_time(self):
        xlabel = "Time [s]"
        ylabel = "Ip [A]"
        state = self.active_state.data.get("equilibrium")

        if state:
            time = state.time[: self.time_idx + 1]
            ip = state.ip[: self.time_idx + 1]
            current_time = state.time[self.time_idx].item()
            title = f"Ip over time, showing t={current_time:.3f} ({len(time)} points)"
        else:
            time, ip, title = [], [], "Waiting for data..."

        return hv.Curve((time, ip), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
        )

    @param.depends("time_idx")
    def plot_f_df_dpsi_profile(self):
        xlabel = "Psi"
        ylabel = "ff'"
        state = self.active_state.data.get("equilibrium")

        if state:
            selected_data = state.isel(time=self.time_idx)
            psi = selected_data.psi
            f_df_dpsi = selected_data.f_df_dpsi
            title = f"ff' profile at t={selected_data.time.item():.3f}"
        else:
            psi, f_df_dpsi, title = [], [], "Waiting for data..."

        return hv.Curve((psi, f_df_dpsi), xlabel, ylabel).opts(
            framewise=True, height=300, width=960, title=title
        )

    @param.depends("time_idx")
    def plot_2d_profile(self):
        xlabel = "r"
        ylabel = "z"
        state = self.active_state.data.get("equilibrium")

        if state:
            selected_data = state.isel(time=self.time_idx)
            f_2d = selected_data.profiles_2d.values.T
            r = selected_data.coords[xlabel]
            z = selected_data.coords[ylabel]
            title = f"Poloidal flux at t={selected_data.time.item():.3f}"
        else:
            r = np.array([0, 1])
            z = np.array([0, 1])
            f_2d = np.full((2, 2), 0)
            title = "Waiting for data..."

        return hv.QuadMesh(
            (r, z, f_2d), kdims=["r_profile", "z_profile"], vdims=["flux"]
        ).opts(
            cmap="viridis",
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=True,
            framewise=True,
            height=300,
            width=960,
            title=title,
        )

    @param.depends("time_idx")
    def plot_ffprime_profile_2d(self):
        state = self.active_state.data.get("equilibrium")
        ylabel = "Time [s]"
        xlabel = "psi"

        if state:
            times = state.time.values[: self.time_idx + 1]
            psi = state.psi.values[self.time_idx]
            f_values_2d = state.f_df_dpsi.values[: self.time_idx + 1, :]
            current_time = state.time[self.time_idx].item()
            title = f"ff' over time, showing t={current_time:.3f}"
        else:
            times = np.array([0, 1])
            psi = np.array([0, 1])
            f_values_2d = np.full((2, 2), 0)
            title = "Waiting for data..."

        return hv.QuadMesh(
            (psi, times, f_values_2d),
            kdims=["profile_dim", "time_dim"],
            vdims=["ffprime"],
        ).opts(
            cmap="viridis",
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=True,
            framewise=True,
            height=300,
            width=960,
            title=title,
        )

    @param.depends("time_idx")
    def plot_coil_currents(self):
        state = self.active_state.data.get("pf_active")
        xlabel = "Time [s]"
        ylabel = "Coil currents [A]"

        if state:
            time = state.time[: self.time_idx]
            current_time = state.time[self.time_idx].item()
            curves = []
            for coil in state.coil.values:
                current = state.currents.sel(coil=coil)[: self.time_idx].values
                curve = hv.Curve((time, current), xlabel, ylabel, label=str(coil)).opts(
                    framewise=True,
                    title=f"coil currents over time, showing t={current_time:.3f}",
                )
                curves.append(curve)
        else:
            curves = [hv.Curve(([0, 1, 2], [0, 1, 2]), xlabel, ylabel)]

        return hv.Overlay(curves).opts(height=600, width=2 * 960)

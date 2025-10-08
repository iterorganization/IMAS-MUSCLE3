import functools
import logging
from enum import Enum
from typing import Dict

import holoviews as hv
import imas
import numpy as np
import panel as pn
import param
import xarray as xr
from imas.ids_data_type import IDSDataType
from imas.ids_primitive import IDSNumericArray
from imas.ids_toplevel import IDSToplevel
from imas.util import tree_iter
from panel.viewable import Viewable

from imas_muscle3.visualization.base import BasePlotter, BaseState
from imas_muscle3.visualization.resizable_float_panel import ResizableFloatPanel

logger = logging.getLogger()


class Dim(Enum):
    """Enum for variable dimensionality."""

    ZERO_D = "0D"
    ONE_D = "1D"
    TWO_D = "2D"


class State(BaseState):
    """Container for simulation state using xarray for time-aligned datasets."""

    discovered_variables = param.Dict(
        default={}, doc="Mapping of IDS name to list of discovered variable paths."
    )
    visualized_variables = param.Dict(
        default={},
        doc="Mapping of IDS name → list of variable paths selected for visualization.",
    )
    variable_dimensions = param.Dict(
        default={},
        doc="Mapping of variable_path → VariableDimension enum ('0D', '1D', or '2D')",
    )
    variable_coord_names = param.Dict(
        default={},
        doc="Mapping of variable_path → coordinates",
    )

    def __init__(self, md_dict: Dict[str, IDSToplevel]) -> None:
        super().__init__(md_dict)
        self._discovery_done = set()

    def _get_coord_name(self, path: str, i: int, coord_obj) -> str:
        if isinstance(coord_obj, IDSNumericArray):
            return coord_obj.metadata.name
        return f"{path}_coord{i}"

    def _discover_variables(self, ids):
        ids_name = ids.metadata.name
        logger.info(f"Discovering float variables in IDS '{ids_name}'...")
        relative_paths = []
        for node in tree_iter(ids, leaf_only=True):
            metadata = node.metadata
            if metadata.data_type != IDSDataType.FLT:
                continue
            if metadata.ndim not in (0, 1, 2):
                continue
            path = str(imas.util.get_full_path(node))
            if path == "time":
                continue
            relative_paths.append(path)
            if metadata.ndim == 0:
                self.variable_dimensions[path] = Dim.ZERO_D
                self.variable_coord_names[path] = []
            elif metadata.ndim == 1:
                if (
                    hasattr(node.coordinates[0], "metadata")
                    and node.coordinates[0].metadata.name == "time"
                ):
                    self.variable_dimensions[path] = Dim.ZERO_D
                    self.variable_coord_names[path] = []
                else:
                    self.variable_dimensions[path] = Dim.ONE_D
                    self.variable_coord_names[path] = [
                        self._get_coord_name(path, 0, node.coordinates[0])
                    ]
            else:
                self.variable_dimensions[path] = Dim.TWO_D
                self.variable_coord_names[path] = [
                    self._get_coord_name(path, 0, node.coordinates[0]),
                    self._get_coord_name(path, 1, node.coordinates[1]),
                ]
        self.discovered_variables[ids_name] = relative_paths
        self._discovery_done.add(ids_name)
        self.param.trigger("discovered_variables")
        self.param.trigger("variable_dimensions")
        self.param.trigger("variable_coord_names")
        logger.info(f"Discovered {len(relative_paths)} variables in IDS '{ids_name}'.")

    def extract(self, ids: IDSToplevel) -> None:
        ids_name = ids.metadata.name
        if ids_name not in self._discovery_done:
            self._discover_variables(ids)
        if (
            ids_name not in self.visualized_variables
            or not self.visualized_variables[ids_name]
        ):
            return
        for name in self.visualized_variables[ids_name]:
            dim = self.variable_dimensions[name]
            if dim == Dim.ZERO_D:
                self._extract_0d(ids, name)
            elif dim == Dim.ONE_D:
                self._extract_1d(ids, name)
            elif dim == Dim.TWO_D:
                self._extract_2d(ids, name)
        self.param.trigger("data")

    def _extract_0d(self, ids, name):
        current_time = float(ids.time[0])
        value_obj = ids[name]
        if value_obj.metadata.ndim == 0:
            value = float(value_obj.value)
        else:
            value = float(value_obj[0])
        new_ds = xr.Dataset(
            {name: ("time", [value])},
            coords={"time": [current_time]},
        )
        if name in self.data:
            self.data[name] = xr.concat([self.data[name], new_ds], dim="time")
        else:
            self.data[name] = new_ds

    def _extract_1d(self, ids, name):
        current_time = float(ids.time[0])
        value_obj = ids[name]
        arr = np.array(value_obj[:], dtype=float)
        coords_obj = value_obj.coordinates[0]
        coord_name = self.variable_coord_names[name][0]
        coords = np.array(coords_obj, dtype=float)

        new_ds = xr.Dataset(
            {
                name: (("time", "i"), arr[np.newaxis, :]),
                f"{name}_{coord_name}": (("time", "i"), coords[np.newaxis, :]),
            },
            coords={"time": [current_time]},
        )
        if name in self.data:
            self.data[name] = xr.concat([self.data[name], new_ds], dim="time")
        else:
            self.data[name] = new_ds

    def _extract_2d(self, ids, name):
        current_time = float(ids.time[0])
        value_obj = ids[name]
        arr = np.array(value_obj[:], dtype=float)
        coords_obj0 = value_obj.coordinates[0]
        coords_obj1 = value_obj.coordinates[1]
        coord_names = self.variable_coord_names[name]
        coords0 = np.array(coords_obj0, dtype=float)
        coords1 = np.array(coords_obj1, dtype=float)

        new_ds = xr.Dataset(
            {
                name: (("time", "y", "x"), arr[np.newaxis, :, :]),
                f"{name}_{coord_names[0]}": (("time", "y"), coords0[np.newaxis, :]),
                f"{name}_{coord_names[1]}": (("time", "x"), coords1[np.newaxis, :]),
            },
            coords={"time": [current_time]},
        )
        if name in self.data:
            self.data[name] = xr.concat([self.data[name], new_ds], dim="time")
        else:
            self.data[name] = new_ds


class Plotter(BasePlotter):
    def __init__(self, state: BaseState) -> None:
        self.ui = pn.Column()
        super().__init__(state)
        self.plot_area = pn.Column(sizing_mode="stretch_width")
        self.variable_selector = pn.widgets.Select(width=400)
        self.add_plot_button = pn.widgets.Button(name="Add Plot", button_type="primary")
        self.add_plot_button.on_click(self._add_plot_callback)

        self.plotting_controls = pn.Row(
            self.variable_selector,
            self.add_plot_button,
            sizing_mode="stretch_width",
            align="center",
        )

        self.filter_input = pn.widgets.TextInput(
            placeholder="Filter...",
            width=400,
        )
        self.filter_input.param.watch(self._update_filter_view, "value_input")
        self.ui.extend([self.filter_input, self.plotting_controls, self.plot_area])

    def _update_filter_view(self, event):
        filter_text = self.filter_input.value_input.lower()
        options = []
        for ids_name, vars_list in self._state.discovered_variables.items():
            for var in vars_list:
                full_name = f"{ids_name}/{var}"
                if not filter_text or filter_text in full_name.lower():
                    options.append(full_name)
        self.variable_selector.options = sorted(options)

    def get_dashboard(self):
        return self.ui

    @param.depends("_state.discovered_variables", watch=True)
    def _update_variable_selector(self) -> None:
        options = []
        for ids_name, vars_list in self._state.discovered_variables.items():
            options.extend(f"{ids_name}/{v}" for v in vars_list)
        self.variable_selector.options = sorted(options)

    def _remove_plot_callback(self, card_to_remove: pn.Card, variable_path: str, event):
        self.plot_area.remove(card_to_remove)
        ids_name, name = variable_path.split("/", 1)
        if ids_name in self._state.visualized_variables:
            new_list = self._state.visualized_variables[ids_name]
            if name in new_list:
                new_list.remove(name)
                self._state.visualized_variables[ids_name] = new_list
        self._state.data.pop(name, None)
        self._state.param.trigger("data")

    def _add_plot_callback(self, event) -> None:
        full_path = self.variable_selector.value
        if not full_path:
            return
        ids_name, name = full_path.split("/", 1)

        if ids_name not in self._state.visualized_variables:
            self._state.visualized_variables[ids_name] = []
        if name in self._state.visualized_variables[ids_name]:
            return
        self._state.visualized_variables[ids_name].append(name)

        plot_func = functools.partial(self._plot_variable_vs_time, name=name)
        dynamic_plot = pn.pane.HoloViews(
            hv.DynamicMap(param.bind(plot_func, time=self.param.time)).opts(
                framewise=True, axiswise=True
            ),
            sizing_mode="stretch_both",
        )
        float_panel = ResizableFloatPanel(dynamic_plot, name=name, contained=False)

        def on_status_change(event):
            if event.new == "closed":
                self._floatpanel_closed_callback(full_path)

        float_panel.param.watch(on_status_change, "status")

        self.plot_area.append(float_panel)

    def _floatpanel_closed_callback(self, variable_path: str, event=None) -> None:
        ids_name, name = variable_path.split("/", 1)
        if ids_name in self._state.visualized_variables:
            new_list = self._state.visualized_variables[ids_name]
            if name in new_list:
                new_list.remove(name)
                self._state.visualized_variables[ids_name] = new_list
        self._state.data.pop(name, None)
        self._state.param.trigger("data")

    def plot_empty(self, name, var_dim):
        if var_dim == Dim.TWO_D:
            empty_vals = np.zeros((1, 1))
            return hv.QuadMesh(
                (np.array([0]), np.array([0]), empty_vals),
                kdims=["x", "y"],
                vdims=[name],
            ).opts(title=f"No data for t = {self.time}", responsive=True)
        return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
            title=f"No data for t = {self.time}", responsive=True
        )

    def plot_1d(self, ds, name, time_index):
        data_var = ds[name].isel(time=time_index).values
        coord_name = self._state.variable_coord_names[name][0]
        coord_var = ds[f"{name}_{coord_name}"].isel(time=time_index).values
        xlabel = coord_name
        ylabel = name
        title = f"{name} (t={float(ds.time.values[time_index]):.3f}s)"
        return hv.Curve((coord_var, data_var), kdims=[xlabel], vdims=[ylabel]).opts(
            title=title, responsive=True
        )

    def plot_2d(self, ds, name, time_index):
        coord_names = self._state.variable_coord_names[name]
        y_name, x_name = coord_names

        data_var = ds[name].isel(time=time_index).values
        x = ds[f"{name}_{x_name}"].isel(time=time_index).values
        y = ds[f"{name}_{y_name}"].isel(time=time_index).values

        title = f"{name} (t={float(ds.time.values[time_index]):.3f}s)"

        return hv.QuadMesh(
            (x, y, data_var),
            kdims=[x_name, y_name],
            vdims=[name],
        ).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            title=title,
            responsive=True,
            xlabel=x_name,
            ylabel=y_name,
        )

    def _plot_variable_vs_time(self, name: str, time: float):
        ds = self.active_state.data.get(name)
        var_dim = self.active_state.variable_dimensions.get(name, Dim.ZERO_D)
        if ds is None or len(ds.time) == 0:
            return self.plot_empty(name, var_dim)

        time_array = ds.time.values
        if time not in time_array:
            return self.plot_empty(name, var_dim)
        time_index = np.where(time_array == time)[0][0]

        if var_dim == Dim.ZERO_D:
            t_vals = time_array[: time_index + 1]
            v_vals = ds[name].isel(time=slice(0, time_index + 1)).values
            return hv.Curve((t_vals, v_vals), kdims=["time"], vdims=[name]).opts(
                title=f"{name} vs time", responsive=True
            )

        if var_dim == Dim.ONE_D:
            return self.plot_1d(ds, name, time_index)

        if var_dim == Dim.TWO_D:
            return self.plot_2d(ds, name, time_index)

    def __panel__(self) -> Viewable:
        return self._panel

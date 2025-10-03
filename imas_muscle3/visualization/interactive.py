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

    def __init__(self, md_dict: Dict[str, IDSToplevel]) -> None:
        super().__init__(md_dict)
        self._discovery_done = set()

    def _discover_variables(self, ids):
        ids_name = ids.metadata.name
        logger.info(f"Discovering float variables in IDS '{ids_name}'...")
        relative_paths_dict = []
        for node in tree_iter(ids, leaf_only=True):
            metadata = node.metadata
            if metadata.data_type == IDSDataType.FLT and metadata.ndim in (0, 1, 2):
                path = str(imas.util.get_full_path(node))
                if path == "time":
                    continue
                relative_paths_dict.append(path)

                if metadata.ndim == 0:
                    self.variable_dimensions[path] = Dim.ZERO_D
                elif metadata.ndim == 1:
                    # If the coordinate is time, treat it as a 0D variable
                    if (
                        hasattr(node.coordinates[0], "metadata")
                        and node.coordinates[0].metadata.name == "time"
                    ):
                        self.variable_dimensions[path] = Dim.ZERO_D
                    else:
                        self.variable_dimensions[path] = Dim.ONE_D
                else:
                    self.variable_dimensions[path] = Dim.TWO_D

        self.discovered_variables[ids_name] = relative_paths_dict
        self._discovery_done.add(ids_name)
        self.param.trigger("discovered_variables")
        self.param.trigger("variable_dimensions")
        logger.info(
            f"Discovered {len(relative_paths_dict)} variables in IDS '{ids_name}'."
        )

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
            # FIXME: use unqiue names instead of "coords"/"dim0"/etc
            if self.variable_dimensions[name] == Dim.ZERO_D:
                self._extract_0d(ids, name)
            elif self.variable_dimensions[name] == Dim.ONE_D:
                self._extract_1d(ids, name)
            elif self.variable_dimensions[name] == Dim.TWO_D:
                self._extract_2d(ids, name)

        self.param.trigger("data")

    def _extract_0d(self, ids, name):
        current_time = ids.time[0]
        value_obj = ids[name]
        if value_obj.metadata.ndim == 0:
            value = value_obj.value
        else:
            value = value_obj[0]
        new_ds = xr.Dataset(
            {name: ("time", [value])},
            coords={"time": [current_time]},
        )
        if name in self.data:
            self.data[name] = xr.concat([self.data[name], new_ds], dim="time")
        else:
            self.data[name] = new_ds

    def _extract_1d(self, ids, name):
        current_time = ids.time[0]
        value_obj = ids[name]
        arr = np.array(value_obj[:], dtype=float)[np.newaxis, :]
        coords_obj = value_obj.coordinates[0]
        if getattr(coords_obj, "is_time_coordinate", False):
            new_ds = xr.Dataset(
                {name: ("time", [arr[0, 0]])},
                coords={"time": [current_time]},
            )
        else:
            if name not in self.data:
                coords = np.array(coords_obj, dtype=float)
                new_ds = xr.Dataset(
                    {name: (("time", "coord"), arr)},
                    coords={"time": [current_time], "coord": coords},
                )
            else:
                existing_ds = self.data[name]
                # FIXME: issue with 2d quadmesh
                if not np.allclose(
                    existing_ds["coord"].values,
                    coords_obj,
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    logger.warning(
                        f"Coordinates for variable {name} differ slightly; using existing coordinates."
                    )
                new_ds = xr.Dataset(
                    {name: (("time", "coord"), arr)},
                    coords={
                        "time": [current_time],
                        "coord": existing_ds["coord"].values,
                    },
                )
        if name in self.data:
            self.data[name] = xr.concat([self.data[name], new_ds], dim="time")
        else:
            self.data[name] = new_ds

    def _extract_2d(self, ids, name):
        current_time = ids.time[0]
        value_obj = ids[name]
        arr = np.array(value_obj[:], dtype=float)[np.newaxis, :, :]
        coords_obj0 = value_obj.coordinates[0]
        coords_obj1 = value_obj.coordinates[1]

        if name not in self.data:
            coords0 = np.array(coords_obj0, dtype=float)
            coords1 = np.array(coords_obj1, dtype=float)
            new_ds = xr.Dataset(
                {name: (("time", "dim0", "dim1"), arr)},
                coords={
                    "time": [current_time],
                    "dim0": coords0,
                    "dim1": coords1,
                },
            )
        else:
            existing_ds = self.data[name]
            new_ds = xr.Dataset(
                {name: (("time", "dim0", "dim1"), arr)},
                coords={
                    "time": [current_time],
                    "dim0": existing_ds["dim0"].values,
                    "dim1": existing_ds["dim1"].values,
                },
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
                kdims=["dim0", "dim1"],
                vdims=[name],
            ).opts(title=f"No data for t = {self.time}", responsive=True)
        else:
            return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
                title=f"No data for t = {self.time}", responsive=True
            )

    def plot_1d(self, data_var, name, time_array, time_index):
        if len(data_var.dims) == 1 or (
            len(data_var.dims) == 2 and data_var.shape[1] == 1
        ):
            value = data_var[: time_index + 1].values.flatten()
            times = time_array[: time_index + 1]
        else:  # 2D profile flattened to 1D
            value = data_var[time_index].values.flatten()
            times = np.arange(value.shape[0])
        title = f"{name} (t={time_array[time_index]:.3f}s)"
        return hv.Curve((times, value), kdims=["time"], vdims=["value"]).opts(
            title=title, xlabel="Time [s]", ylabel=name, responsive=True
        )

    def plot_2d(self, data_var, name, time_array, time_index):
        if len(data_var.dims) == 1:
            vals = data_var[: time_index + 1].values[np.newaxis, :]
            x = np.arange(vals.shape[1])
            y = np.zeros(1)
        elif len(data_var.dims) == 2:
            vals = data_var[time_index].values
            x = ds["dim0"].values
            y = ds["dim1"].values
        elif len(data_var.dims) == 3:
            vals = data_var[time_index].values.T
            x = ds["dim0"].values
            y = ds["dim1"].values
        return hv.QuadMesh((x, y, vals), kdims=["dim0", "dim1"], vdims=[name]).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            title=f"{name} 2D (t={time_array[time_index]:.3f}s)",
            responsive=True,
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
        data_var = ds[name]

        if var_dim == Dim.ZERO_D or var_dim == Dim.ONE_D:
            return self.plot_1d(data_var, name, time_array, time_index)
        elif var_dim == Dim.TWO_D:
            return self.plot_2d(data_var, name, time_array, time_index)

    def __panel__(self) -> Viewable:
        return self._panel

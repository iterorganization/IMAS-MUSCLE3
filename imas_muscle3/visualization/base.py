import functools
import logging
from enum import Enum
from typing import Dict, List

import holoviews as hv
import numpy as np
import panel as pn
import param
import xarray as xr
from imas.ids_data_type import IDSDataType
from imas.ids_toplevel import IDSToplevel
from imas.util import tree_iter
from panel.viewable import Viewable, Viewer

from imas_muscle3.visualization.resizable_float_panel import ResizableFloatPanel

logger = logging.getLogger()


class VariableDimension(Enum):
    """Enum for variable dimensionality."""

    ZERO_D = "0D"
    ONE_D = "1D"
    TWO_D = "2D"


class BaseState(param.Parameterized):
    """Container for simulation state using xarray for time-aligned datasets."""

    data = param.Dict(
        default={}, doc="Dictionary to store xarray Datasets of visualized variables."
    )
    md = param.Dict(
        default={}, doc="Dictionary of IDS name → machine description data objects."
    )
    discovered_variables = param.Dict(
        default={}, doc="Mapping of IDS name → list of discovered variable paths."
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
        super().__init__()
        self.md = md_dict
        self._discovery_done = set()

    def _discover_variables(self, ids):
        ids_name = ids.metadata.name
        logger.info(f"Discovering float variables in IDS '{ids_name}'...")
        relative_paths_dict = []
        for node in tree_iter(ids, leaf_only=True):
            metadata = node.metadata
            if metadata.data_type == IDSDataType.FLT and metadata.ndim in (0, 1, 2):
                path_str = str(node._path)
                relative_paths_dict.append(path_str)

                # Determine variable dimension during discovery
                if metadata.ndim == 0:
                    self.variable_dimensions[path_str] = VariableDimension.ZERO_D
                elif metadata.ndim == 1:
                    self.variable_dimensions[path_str] = VariableDimension.ONE_D
                else:  # metadata.ndim == 2
                    self.variable_dimensions[path_str] = VariableDimension.TWO_D

        self.discovered_variables[ids_name] = relative_paths_dict
        self._discovery_done.add(ids_name)
        self.param.trigger("discovered_variables")
        self.param.trigger("variable_dimensions")
        logger.info(
            f"Discovered {len(relative_paths_dict)} variables in IDS '{ids_name}'."
        )

    def extract(self, ids: IDSToplevel) -> None:
        ids_name = ids.metadata.name
        current_time = ids.time[0]

        if ids_name not in self._discovery_done:
            self._discover_variables(ids)

        if (
            ids_name not in self.visualized_variables
            or not self.visualized_variables[ids_name]
        ):
            return

        for path_str in self.visualized_variables[ids_name]:
            value_obj = ids[path_str]

            if value_obj.metadata.ndim == 0:
                new_ds = xr.Dataset(
                    {path_str: ("time", [value_obj.value])},
                    coords={"time": [current_time]},
                )
                if path_str in self.data:
                    self.data[path_str] = xr.concat(
                        [self.data[path_str], new_ds], dim="time"
                    )
                else:
                    self.data[path_str] = new_ds

            elif value_obj.metadata.ndim == 1:
                arr = np.array(value_obj[:], dtype=float)[np.newaxis, :]
                coords_obj = value_obj.coordinates[0]
                if getattr(coords_obj, "is_time_coordinate", False):
                    new_ds = xr.Dataset(
                        {path_str: ("time", [arr[0, 0]])},
                        coords={"time": [current_time]},
                    )
                else:
                    if path_str not in self.data:
                        coords = np.array(coords_obj, dtype=float)
                        new_ds = xr.Dataset(
                            {path_str: (("time", "coord"), arr)},
                            coords={"time": [current_time], "coord": coords},
                        )
                    else:
                        existing_ds = self.data[path_str]
                        if not np.allclose(
                            existing_ds["coord"].values,
                            coords_obj,
                            rtol=1e-6,
                            atol=1e-8,
                        ):
                            logger.warning(
                                f"Coordinates for variable {path_str} differ slightly; using existing coordinates."
                            )
                        new_ds = xr.Dataset(
                            {path_str: (("time", "coord"), arr)},
                            coords={
                                "time": [current_time],
                                "coord": existing_ds["coord"].values,
                            },
                        )
                if path_str in self.data:
                    self.data[path_str] = xr.concat(
                        [self.data[path_str], new_ds], dim="time"
                    )
                else:
                    self.data[path_str] = new_ds

            elif value_obj.metadata.ndim == 2:
                arr = np.array(value_obj[:], dtype=float)[np.newaxis, :, :]
                coords_obj0 = value_obj.coordinates[0]
                coords_obj1 = value_obj.coordinates[1]

                if path_str not in self.data:
                    coords0 = np.array(coords_obj0, dtype=float)
                    coords1 = np.array(coords_obj1, dtype=float)
                    new_ds = xr.Dataset(
                        {path_str: (("time", "dim0", "dim1"), arr)},
                        coords={
                            "time": [current_time],
                            "dim0": coords0,
                            "dim1": coords1,
                        },
                    )
                else:
                    existing_ds = self.data[path_str]
                    new_ds = xr.Dataset(
                        {path_str: (("time", "dim0", "dim1"), arr)},
                        coords={
                            "time": [current_time],
                            "dim0": existing_ds["dim0"].values,
                            "dim1": existing_ds["dim1"].values,
                        },
                    )
                if path_str in self.data:
                    self.data[path_str] = xr.concat(
                        [self.data[path_str], new_ds], dim="time"
                    )
                else:
                    self.data[path_str] = new_ds

        self.param.trigger("data")


class BasePlotter(Viewer):
    _state = param.ClassSelector(class_=BaseState)
    _live_view = param.Boolean(default=True)
    time = param.Number(default=0.0)

    def __init__(self, state: BaseState) -> None:
        super().__init__(_state=state)
        self.active_state = self._state
        self.plot_area = pn.Column(sizing_mode="stretch_width")

        self.live_view_checkbox = pn.widgets.Checkbox.from_param(
            self.param._live_view, align="center"
        )
        self.time_slider_widget = pn.widgets.DiscretePlayer.from_param(
            self.param.time,
            margin=15,
            interval=100,
            options=[0.0],
            value=0.0,
            visible=self.param._live_view.rx.not_(),
        )
        self.time_label = pn.pane.Markdown("", align="center")
        self.variable_selector = pn.widgets.Select(
            name="Variable to Plot",
            options=self._get_all_discovered_vars(),
            width=400,
        )
        self.add_plot_button = pn.widgets.Button(name="Add Plot", button_type="primary")
        self.add_plot_button.on_click(self._add_plot_callback)

        playback_controls = pn.Row(
            self.live_view_checkbox,
            self.time_slider_widget,
            self.time_label,
            sizing_mode="stretch_width",
        )
        plotting_controls = pn.Row(
            self.variable_selector,
            self.add_plot_button,
            sizing_mode="stretch_width",
            align="center",
        )

        self._panel = pn.Column(
            playback_controls,
            plotting_controls,
            self.plot_area,
            sizing_mode="stretch_width",
        )

    def _get_all_discovered_vars(self) -> List[str]:
        options = []
        for ids_name, vars_list in self._state.discovered_variables.items():
            options.extend(f"{ids_name}/{v}" for v in vars_list)
        return sorted(options)

    @param.depends("_state.discovered_variables", watch=True)
    def _update_variable_selector(self) -> None:
        self.variable_selector.options = self._get_all_discovered_vars()

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

        plot_func = functools.partial(self.plot_variable_vs_time, variable_path=name)
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

    @param.depends("time", watch=True)
    def update_time_label(self) -> None:
        if not self.active_state.data:
            self.time_label.object = "### t = N/A"
            return
        time_data = None
        for vars_list in self._state.visualized_variables.values():
            for path_str in vars_list:
                if path_str in self.active_state.data:
                    time_data = self.active_state.data[path_str].time.values
                    break
            if time_data is not None:
                break
        if time_data is not None and self.time in time_data:
            self.time_label.object = f"### t = {self.time:.5e} s"
        else:
            self.time_label.object = "### t = N/A"

    @param.depends("_state.data", watch=True)
    def _update_on_new_data(self) -> None:
        if not self._state.data:
            return
        all_times = sorted(
            set(np.concatenate([d.time.values for d in self._state.data.values()]))
        )
        if not all_times:
            return
        self.time_slider_widget.options = list(all_times)
        if self._live_view:
            self.active_state = self._state
            self.time = all_times[-1]

    def plot_variable_vs_time(self, variable_path: str, time: float):
        ds = self.active_state.data.get(variable_path)
        var_dim = self.active_state.variable_dimensions.get(
            variable_path, VariableDimension.ZERO_D
        )

        if ds is None or len(ds.time) == 0:
            if var_dim == VariableDimension.TWO_D:
                empty_vals = np.zeros((1, 1))
                return hv.QuadMesh(
                    (np.array([0]), np.array([0]), empty_vals),
                    kdims=["dim0", "dim1"],
                    vdims=[variable_path],
                ).opts(title="Waiting for data...", responsive=True)
            else:
                return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
                    title="Waiting for data...", responsive=True
                )

        time_array = ds.time.values
        if time not in time_array:
            if var_dim == VariableDimension.TWO_D:
                empty_vals = np.zeros((1, 1))
                return hv.QuadMesh(
                    (np.array([0]), np.array([0]), empty_vals),
                    kdims=["dim0", "dim1"],
                    vdims=[variable_path],
                ).opts(title="No data for selected time", responsive=True)
            else:
                return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
                    title="No data for selected time", responsive=True
                )

        idx = np.where(time_array == time)[0][0]
        data_var = ds[variable_path]

        if var_dim == VariableDimension.ZERO_D or var_dim == VariableDimension.ONE_D:
            if len(data_var.dims) == 1 or (
                len(data_var.dims) == 2 and data_var.shape[1] == 1
            ):
                value = data_var[: idx + 1].values.flatten()
                times = time_array[: idx + 1]
            else:  # 2D profile flattened to 1D
                value = data_var[idx].values.flatten()
                times = np.arange(value.shape[0])
            title = f"{variable_path} (t={time_array[idx]:.3f}s)"
            return hv.Curve((times, value), kdims=["time"], vdims=["value"]).opts(
                title=title, xlabel="Time [s]", ylabel=variable_path, responsive=True
            )

        else:  # TWO_D
            if len(data_var.dims) == 1:
                vals = data_var[: idx + 1].values[np.newaxis, :]
                x = np.arange(vals.shape[1])
                y = np.zeros(1)
            elif len(data_var.dims) == 2:
                vals = data_var[idx].values
                x = ds["dim0"].values
                y = ds["dim1"].values
            elif len(data_var.dims) == 3:
                vals = data_var[idx].values.T
                x = ds["dim0"].values
                y = ds["dim1"].values
            return hv.QuadMesh(
                (x, y, vals), kdims=["dim0", "dim1"], vdims=[variable_path]
            ).opts(
                cmap="viridis",
                colorbar=True,
                framewise=True,
                title=f"{variable_path} 2D (t={time_array[idx]:.3f}s)",
                responsive=True,
            )

    def __panel__(self) -> Viewable:
        return self._panel

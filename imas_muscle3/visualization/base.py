import functools
import logging
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

logger = logging.getLogger()


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

    def __init__(self, md_dict: Dict[str, IDSToplevel]) -> None:
        super().__init__()
        self.md = md_dict
        self._discovery_done = set()

    def extract(self, ids: IDSToplevel) -> None:
        ids_name = ids.metadata.name
        current_time = ids.time[0]

        # Discover variables per IDS
        if ids_name not in self._discovery_done:
            logger.info(f"Discovering float variables in IDS '{ids_name}'...")
            relative_paths_dict = []
            for node in tree_iter(ids, leaf_only=True):
                metadata = node.metadata
                if metadata.data_type == IDSDataType.FLT and metadata.ndim in (0, 1):
                    relative_paths_dict.append(str(node._path))
            self.discovered_variables[ids_name] = relative_paths_dict
            self._discovery_done.add(ids_name)
            self.param.trigger("discovered_variables")
            logger.info(
                f"Discovered {len(relative_paths_dict)} variables in IDS '{ids_name}'."
            )

        # No variables selected for visualization
        if (
            ids_name not in self.visualized_variables
            or not self.visualized_variables[ids_name]
        ):
            return

        for path_str in self.visualized_variables[ids_name]:
            value_obj = ids[path_str]

            # 0D variable
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

            # 1D variable
            else:
                arr = np.array(value_obj[:], dtype=float)[np.newaxis, :]
                coords_obj = value_obj.coordinates[0]
                if getattr(coords_obj, "is_time_coordinate", False):
                    # treat like 0D
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

        self.param.trigger("data")


class BasePlotter(Viewer):
    _state = param.ClassSelector(class_=BaseState)
    _live_view = param.Boolean(default=True)
    time_index = param.Integer(default=0)

    def __init__(self, state: BaseState) -> None:
        super().__init__(_state=state)
        self.active_state = self._state
        self.plot_area = pn.Column(sizing_mode="stretch_width")

        self.live_view_checkbox = pn.widgets.Checkbox.from_param(
            self.param._live_view, align="center"
        )
        self.time_slider_widget = pn.widgets.DiscretePlayer.from_param(
            self.param.time_index,
            margin=15,
            interval=100,
            options=[0],
            value=0,
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
            options.extend(f"{ids_name}.{v}" for v in vars_list)
        return sorted(options)

    @param.depends("_state.discovered_variables", watch=True)
    def _update_variable_selector(self) -> None:
        self.variable_selector.options = self._get_all_discovered_vars()

    def _remove_plot_callback(self, card_to_remove: pn.Card, variable_path: str, event):
        self.plot_area.remove(card_to_remove)
        ids_name, path_str = variable_path.split(".", 1)
        if ids_name in self._state.visualized_variables:
            new_list = self._state.visualized_variables[ids_name]
            if path_str in new_list:
                new_list.remove(path_str)
                self._state.visualized_variables[ids_name] = new_list
        self._state.data.pop(path_str, None)
        self._state.param.trigger("data")

    def _add_plot_callback(self, event) -> None:
        full_path = self.variable_selector.value
        if not full_path:
            return
        ids_name, path_str = full_path.split(".", 1)
        if ids_name not in self._state.visualized_variables:
            self._state.visualized_variables[ids_name] = []
        if path_str in self._state.visualized_variables[ids_name]:
            return
        self._state.visualized_variables[ids_name].append(path_str)

        plot_func = functools.partial(
            self.plot_variable_vs_time, variable_path=path_str
        )
        dynamic_plot = hv.DynamicMap(
            param.bind(plot_func, time_index=self.param.time_index)
        ).opts(framewise=True, axiswise=True)

        remove_button = pn.widgets.Button(name="Remove", button_type="danger", width=80)
        plot_card = pn.Card(
            dynamic_plot,
            header=full_path,
            collapsible=True,
            sizing_mode="stretch_width",
        )
        remove_button.on_click(
            functools.partial(self._remove_plot_callback, plot_card, full_path)
        )

        plot_card.header = pn.Row(
            pn.pane.Markdown(f"**{full_path}**"),
            remove_button,
            align="center",
            sizing_mode="stretch_width",
        )

        self.plot_area.append(plot_card)

    @param.depends("time_index", watch=True)
    def update_time_label(self) -> None:
        if not self.active_state.data:
            self.time_label.object = "### t = N/A"
            return
        time_data = None
        for vars_list in self._state.visualized_variables.values():
            for path_str in vars_list:
                if path_str in self.active_state.data:
                    time_data = self.active_state.data[path_str].time
                    break
            if time_data is not None:
                break
        if time_data is not None and self.time_index < len(time_data):
            t = float(time_data[self.time_index])
            self.time_label.object = f"### t = {t:.5e} s"
        else:
            self.time_label.object = "### t = N/A"

    @param.depends("_state.data", watch=True)
    def _update_on_new_data(self) -> None:
        if not self._state.data:
            return
        num_steps = max(
            len(d.time) for d in self._state.data.values() if len(d.time) > 0
        )
        if num_steps == 0:
            return
        self.time_slider_widget.options = list(range(num_steps))
        if self._live_view:
            self.active_state = self._state
            self.time_index = num_steps - 1

    def plot_variable_vs_time(self, variable_path: str, time_index: int):
        ds = self.active_state.data.get(variable_path)
        if ds is None or len(ds.time) == 0:
            return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
                title="Waiting for data...", height=300, width=960
            )

        idx = min(time_index, len(ds.time) - 1)
        time = ds.time[: idx + 1].values

        data_var = ds[variable_path]
        # Determine if variable is effectively 0D over time (single coord or time-coordinate)
        if len(data_var.dims) == 1 or (
            len(data_var.dims) == 2 and data_var.shape[1] == 1
        ):
            # flatten values
            value = data_var[: idx + 1].values.flatten()
            title = f"{variable_path} (t = {time[-1]:.3f}s)"
            return hv.Curve((time, value), kdims=["time"], vdims=["value"]).opts(
                height=300,
                width=960,
                title=title,
                xlabel="Time [s]",
                ylabel=variable_path,
            )
        else:
            coords = ds["coord"].values
            profile = data_var[idx].values
            title = f"{variable_path} profile (t = {float(ds.time[idx]):.3f}s)"
            return hv.Curve((coords, profile), kdims=["coord"], vdims=["value"]).opts(
                height=300,
                width=960,
                title=title,
                xlabel="Coordinate",
                ylabel=variable_path,
            )

    def __panel__(self) -> Viewable:
        return self._panel

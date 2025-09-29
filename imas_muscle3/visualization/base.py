import functools
import logging
from typing import Dict

import holoviews as hv
import numpy as np
import panel as pn
import param
from imas.ids_data_type import IDSDataType
from imas.ids_toplevel import IDSToplevel
from imas.util import tree_iter
from panel.viewable import Viewable, Viewer

logger = logging.getLogger()


class BaseState(param.Parameterized):
    """Abstract container for simulation state."""

    data = param.Dict(
        default={}, doc="Dictionary to store time series of discovered variables."
    )
    md = param.Dict(
        default={},
        doc="Dictionary of IDS name → machine description data objects.",
    )
    discovered_variables = param.Dict(
        default={}, doc="Mapping of string paths to IMAS path tuples."
    )

    def __init__(self, md_dict: Dict[str, IDSToplevel]) -> None:
        super().__init__()
        self.md = md_dict
        self._discovery_done = False

    def _find_float0d_paths(self, ids: IDSToplevel) -> dict:
        """Find all 0D floats and return a dict mapping string paths to path tuples."""
        paths = {}
        for node in tree_iter(ids, leaf_only=True):
            metadata = node.metadata
            if metadata.data_type == IDSDataType.FLT and metadata.ndim == 0:
                path_tuple = node._path
                path_str = str(node._path)
                paths[path_str] = path_tuple
        return paths

    def extract(self, ids: IDSToplevel) -> None:
        """Discover 0D float variables on first run and extract data."""
        ids_name = ids.metadata.name
        current_time = ids.time[0]

        if not self._discovery_done:
            logger.info("First IDS received, discovering 0D float variables...")
            relative_paths_dict = self._find_float0d_paths(ids)

            full_paths_dict = {
                f"{ids_name}.{path_str}": path_tuple
                for path_str, path_tuple in relative_paths_dict.items()
            }

            self.discovered_variables.update(full_paths_dict)
            print(self.discovered_variables)
            self.param.trigger("discovered_variables")
            logger.info(f"Discovered {len(full_paths_dict)} variables.")
            if full_paths_dict:
                self._discovery_done = True

        for path_str, path_tuple in self.discovered_variables.items():
            value_obj = ids[path_tuple]

            if path_str not in self.data:
                self.data[path_str] = {"time": [], "value": []}

            self.data[path_str]["time"].append(current_time)
            self.data[path_str]["value"].append(value_obj.value)

        self.param.trigger("data")


class BasePlotter(Viewer):
    _state = param.ClassSelector(
        class_=BaseState,
        doc="The state object containing the data from the simulation.",
    )
    _live_view = param.Boolean(
        default=True,
        label="Live View",
        doc="Flag for setting UI to live view mode",
    )
    time_index = param.Integer(
        default=0,
        label="Time Step",
        doc="Currently selected time index in the DiscretePlayer",
    )

    def __init__(self, state: BaseState) -> None:
        super().__init__(_state=state)
        self._frozen_state = None
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
            options=list(self._state.discovered_variables.keys()),
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

    @param.depends("_state.discovered_variables", watch=True)
    def _update_variable_selector(self) -> None:
        """Update selector with the string keys from the discovery dictionary."""
        self.variable_selector.options = list(self._state.discovered_variables)

    def _add_plot_callback(self, event) -> None:
        selected_var = self.variable_selector.value
        if not selected_var or selected_var in [p.name for p in self.plot_area]:
            return

        plot_func = functools.partial(
            self.plot_variable_vs_time, variable_path=selected_var
        )
        dynamic_plot = hv.DynamicMap(
            param.bind(plot_func, time_index=self.param.time_index)
        )
        self.plot_area.append(dynamic_plot)

    def plot_variable_vs_time(self, time_index: int, variable_path: str):
        xlabel = "Time [s]"
        ylabel = variable_path
        state_data = self.active_state.data.get(variable_path)

        if state_data:
            time = state_data["time"][: time_index + 1]
            value = state_data["value"][: time_index + 1]
            title = f"{variable_path} over time"
        else:
            time, value, title = [], [], "Waiting for data..."

        return hv.Curve((time, value), kdims=["time"], vdims=["value"]).opts(
            framewise=True,
            height=300,
            width=960,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    @param.depends("_live_view", watch=True)
    def _store_frozen_state(self) -> None:
        if self._live_view:
            self._frozen_state = None
            self.active_state = self._state
        else:

            class FrozenState:
                def __init__(self, data):
                    self.data = data

            self._frozen_state = FrozenState(self.active_state.data.copy())
            self.active_state = self._frozen_state

    @param.depends("time_index", watch=True)
    def update_time_label(self) -> None:
        if not self.active_state.data:
            return

        first_var_path = next(iter(self.active_state.data))
        time_data = self.active_state.data[first_var_path]["time"]

        if self.time_index < len(time_data):
            t = time_data[self.time_index]
            self.time_label.object = f"### t = {t:.5e} s"

    @param.depends("_state.data", watch=True)
    def _update_on_new_data(self) -> None:
        print(self._state.data)
        if not self._state.data:
            return

        first_var_path = next(iter(self._state.data))
        num_steps = len(self._state.data[first_var_path]["time"])

        self.time_slider_widget.options = list(range(num_steps))

        if self._live_view:
            self.active_state = self._state
            self.time_index = num_steps - 1
            if self.time_index == 0:
                self.param.trigger("time_index")
        else:

            class FrozenState:
                def __init__(self, data):
                    self.data = data

            self._frozen_state = FrozenState(self._state.data.copy())
            self.active_state = self._frozen_state

    def __panel__(self) -> Viewable:
        return self._panel

import functools
import logging
from typing import Dict, List

import holoviews as hv
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
        default={}, doc="Dictionary to store time series of visualized variables."
    )
    md = param.Dict(
        default={},
        doc="Dictionary of IDS name → machine description data objects.",
    )
    discovered_variables = param.Dict(
        default={}, doc="Mapping of string paths to IMAS path tuples."
    )
    visualized_variables = param.List(
        default=[], doc="List of variable paths selected for visualization."
    )

    def __init__(self, md_dict: Dict[str, IDSToplevel]) -> None:
        super().__init__()
        self.md = md_dict
        self._discovery_done = False

    def extract(self, ids: IDSToplevel) -> None:
        ids_name = ids.metadata.name
        current_time = ids.time[0]

        if not self._discovery_done:
            logger.info("First IDS received, discovering float variables...")
            relative_paths_dict = {}
            for node in tree_iter(ids, leaf_only=True):
                metadata = node.metadata
                if metadata.data_type == IDSDataType.FLT and metadata.ndim in (0, 1):
                    path = node._path
                    path_str = str(path)
                    relative_paths_dict[path_str] = path

            full_paths_dict = {
                f"{ids_name}.{path_str}": path
                for path_str, path in relative_paths_dict.items()
            }
            self.discovered_variables.update(full_paths_dict)
            self.param.trigger("discovered_variables")
            logger.info(f"Discovered {len(full_paths_dict)} variables.")
            if full_paths_dict:
                self._discovery_done = True

        if not self.visualized_variables:
            return

        for path_str in self.visualized_variables:
            path = self.discovered_variables.get(path_str)
            value_obj = ids[path]

            if path_str not in self.data:
                self.data[path_str] = {"time": [], "value": [], "coords": None}

            if value_obj.metadata.ndim == 0:
                self.data[path_str]["time"].append(current_time)
                self.data[path_str]["value"].append(value_obj.value)

            elif value_obj.metadata.ndim == 1:
                # Store profile values and coordinates
                self.data[path_str]["time"].append(current_time)
                self.data[path_str]["value"].append(value_obj[:])  # ndarray
                if self.data[path_str]["coords"] is None:
                    self.data[path_str]["coords"] = value_obj.coordinates[0][:]

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
            options=sorted(list(self._state.discovered_variables.keys())),
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
        self.variable_selector.options = sorted(list(self._state.discovered_variables))

    def _remove_plot_callback(
        self, card_to_remove: pn.Card, variable_path: str, event
    ) -> None:
        """Callback to remove a plot and stop tracking its data."""
        self.plot_area.remove(card_to_remove)

        # Stop collecting new data
        if variable_path in self._state.visualized_variables:
            new_list = self._state.visualized_variables
            new_list.remove(variable_path)
            self._state.visualized_variables = new_list

        # Remove historical data
        self._state.data.pop(variable_path, None)
        self._state.param.trigger("data")  # Notify watchers data changed

    def _add_plot_callback(self, event) -> None:
        """Adds a new plot and tells the state to start collecting data for it."""
        selected_var = self.variable_selector.value
        if not selected_var or selected_var in self._state.visualized_variables:
            return

        # --- MODIFIED: Tell state to start tracking this variable ---
        self._state.visualized_variables = self._state.visualized_variables + [
            selected_var
        ]

        plot_func = functools.partial(
            self.plot_variable_vs_time, variable_path=selected_var
        )
        dynamic_plot = hv.DynamicMap(
            param.bind(plot_func, time_index=self.param.time_index)
        ).opts(framewise=True, axiswise=True)

        remove_button = pn.widgets.Button(name="Remove", button_type="danger", width=80)

        plot_card = pn.Card(
            dynamic_plot,
            header=selected_var,
            collapsible=True,
            sizing_mode="stretch_width",
        )
        remove_button.on_click(
            functools.partial(self._remove_plot_callback, plot_card, selected_var)
        )

        plot_card.header = pn.Row(
            pn.pane.Markdown(f"**{selected_var}**"),
            remove_button,
            align="center",
            sizing_mode="stretch_width",
        )

        self.plot_area.append(plot_card)

    def plot_variable_vs_time(self, time_index: int, variable_path: str):
        state_data = self.active_state.data.get(variable_path)

        if not state_data or not state_data["time"]:
            return hv.Curve(([], []), kdims=["x"], vdims=["value"]).opts(
                title="Waiting for data...", height=300, width=960
            )

        times = state_data["time"]
        values = state_data["value"]

        if isinstance(values[0], (float, int)):  # 0D case
            time = times[: time_index + 1]
            value = values[: time_index + 1]
            title = f"{variable_path} (t = {time[-1]:.3f}s)"
            return hv.Curve((time, value), kdims=["time"], vdims=["value"]).opts(
                height=300,
                width=960,
                title=title,
                xlabel="Time [s]",
                ylabel=variable_path,
            )

        else:  # 1D case
            coords = state_data["coords"]
            profile = values[time_index]
            t = times[time_index]
            title = f"{variable_path} profile (t = {t:.3f}s)"
            return hv.Curve((coords, profile), kdims=["coord"], vdims=["value"]).opts(
                height=300,
                width=960,
                title=title,
                xlabel="Coordinate",
                ylabel=variable_path,
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
            self.time_label.object = "### t = N/A"
            return

        # Try to get time data from any of the visualized variables
        time_data = []
        for var in self._state.visualized_variables:
            if var in self.active_state.data and self.active_state.data[var]["time"]:
                time_data = self.active_state.data[var]["time"]
                break

        if time_data and self.time_index < len(time_data):
            t = time_data[self.time_index]
            self.time_label.object = f"### t = {t:.5e} s"
        else:
            self.time_label.object = "### t = N/A"

    @param.depends("_state.data", watch=True)
    def _update_on_new_data(self) -> None:
        if not self._state.data:
            return

        # Find the max number of steps among all visualized variables
        num_steps = 0
        if self._state.data:
            num_steps = max(
                (len(d["time"]) for d in self._state.data.values() if d["time"]),
                default=0,
            )

        if num_steps == 0:
            return

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

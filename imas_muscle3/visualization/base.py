from typing import Dict

import numpy as np
import panel as pn
import param
from imas.ids_toplevel import IDSToplevel
from panel.viewable import Viewable, Viewer


class BaseState(param.Parameterized):
    """Abstract container for simulation state. Holds live simulation data
    as well as data from a machine description.
    """

    data = param.Dict(
        default={}, doc="Mapping of IDS name to live IDS data objects."
    )
    md = param.Dict(
        default={},
        doc="Mapping of IDS name to machine description data objects.",
    )

    def __init__(
        self, md_dict: Dict[str, IDSToplevel], extract_all: bool = False
    ) -> None:
        super().__init__()
        self.extract_all = extract_all
        self.md = md_dict

    def extract(self, ids: IDSToplevel) -> None:
        """Extract data from an IDS and store it into a state object. Must be
        implemented by subclasses.

        Args:
            ids: An IDS containing simulation results.
        """
        raise NotImplementedError(
            "A state class needs to implement an `extract` method"
        )


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
    time = param.Number(
        default=0.0,
        label="Time Step",
        doc="Currently selected time step in the DiscretePlayer",
    )

    def __init__(self, state: BaseState) -> None:
        super().__init__(_state=state)
        self._frozen_state = None
        self.active_state = self._state

        self.live_view_checkbox = pn.widgets.Checkbox.from_param(
            self.param._live_view
        )
        self.time_slider_widget = pn.widgets.DiscretePlayer.from_param(
            self.param.time,
            margin=15,
            interval=100,
            options=[0.0],
            value=0.0,
            visible=self.param._live_view.rx.not_(),
        )
        self.time_label = pn.pane.Markdown("")  # type: ignore[no-untyped-call]
        controls = pn.Row(
            self.live_view_checkbox, self.time_slider_widget, self.time_label
        )
        plots = self.get_dashboard()
        self._panel = pn.Column(controls, plots)

    def get_dashboard(self) -> Viewable:
        """Return Panel layout for the visualization."""
        raise NotImplementedError(
            "a plotter class needs to implement a `get_dashboard` method"
        )

    @param.depends("_live_view", watch=True)  # type: ignore[misc]
    def _store_frozen_state(self) -> None:
        """Store frozen state when live view is toggled."""
        if self._live_view:
            self._frozen_state = None
        else:
            self._frozen_state = self._state

    @param.depends("time", watch=True)  # type: ignore[misc]
    def update_time_label(self) -> None:
        self.time_label.object = f"## showing t = {self.time:.5e} s"

    @param.depends("_state.data", watch=True)  # type: ignore[misc]
    def _update_on_new_data(self) -> None:
        if not self._state.data:
            return
        all_times = sorted(
            set(
                np.concatenate(
                    [d.time.values for d in self._state.data.values()]
                )
            )
        )
        if not all_times:
            return
        self.time_slider_widget.options = list(all_times)
        if self._live_view:
            self.active_state = self._state
            self.time = all_times[-1]

    def __panel__(self) -> Viewable:
        return self._panel

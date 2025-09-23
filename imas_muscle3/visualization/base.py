import panel as pn
import param
from panel.viewable import Viewer


class BaseState(param.Parameterized):
    """Abstract container for simulation state. Holds live simulation data as well as
    data from a machine description.
    """

    data = param.Dict(default={}, doc="Dictionary of IDS name → live IDS data objects.")
    md = param.Dict(
        default={}, doc="Dictionary of IDS name → machine description data objects."
    )

    def __init__(self, md_dict, **kwargs):
        super().__init__(**kwargs)
        self.md = md_dict

    def extract(self, ids):
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
    time_index = param.Integer(
        default=0,
        label="Time Step",
        doc="Currently selected time index in the DiscretePlayer",
    )

    def __init__(self, state, **params):
        super().__init__(_state=state, **params)
        self._frozen_state = None
        self.active_state = self._state

        self.live_view_checkbox = pn.widgets.Checkbox.from_param(self.param._live_view)
        self.time_slider_widget = pn.widgets.DiscretePlayer.from_param(
            self.param.time_index,
            margin=15,
            interval=10,
            options=[0],
            value=0,
            visible=self.param._live_view.rx.not_(),
        )
        self.time_label = pn.pane.Markdown("")
        controls = pn.Row(
            self.live_view_checkbox, self.time_slider_widget, self.time_label
        )
        plots = self.get_dashboard()
        self._panel = pn.Column(controls, plots)

    def get_dashboard(self):
        """Return Panel layout for the visualization."""
        raise NotImplementedError(
            "a plotter class needs to implement a `get_dashboard` method"
        )

    @param.depends("_live_view", watch=True)
    def _store_frozen_state(self):
        """Store frozen state when live view is toggled."""
        if self._live_view:
            self._frozen_state = None
        else:
            self._frozen_state = self._state

    @param.depends("time_index", watch=True)
    def update_time_label(self):
        t = (
            self.active_state.data[next(iter(self.active_state.data))]
            .time[self.time_index]
            .item()
        )
        self.time_label.object = f"# showing t = {t:.5e} s"

    @param.depends("_state.data", watch=True)
    def _update_on_new_data(self):
        """Update time slider and time index when new data arrives."""
        state_data = next(iter(self._state.data.values()), None)
        if not state_data:
            return
        if not hasattr(state_data, "time"):
            raise KeyError("The state must contain a time coordinate")
        num_steps = len(state_data.time)

        self.time_slider_widget.options = list(range(num_steps))
        if self._live_view:
            self.active_state = self._state
            self.time_index = num_steps - 1
            # Ensure it triggers on first time step
            if self.time_index == 0:
                self.param.trigger("time_index")
        else:
            self.active_state = self._frozen_state

    def __panel__(self):
        return self._panel

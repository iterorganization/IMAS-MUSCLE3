import panel as pn
import param
from panel.viewable import Viewer


class BaseState(param.Parameterized):
    data = param.Dict(default={})
    md = param.Dict(default={})

    def __init__(self, md_dict, **kwargs):
        super().__init__(**kwargs)
        self.md = md_dict

    def extract(self, ids):
        raise NotImplementedError(
            "a state class needs to implement an `extract` method"
        )


class BasePlotter(Viewer):
    state = param.Parameter(doc="The live state object from the simulation.")
    time_idx = param.Integer(default=0, label="Time Step")
    live_view = param.Boolean(default=True, label="Live View")

    def __init__(self, state, **params):
        super().__init__(state=state, **params)
        self._frozen_state = None
        self.active_state = self.state

    def __panel__(self):
        self.time_slider_widget = pn.widgets.DiscretePlayer.from_param(
            self.param.time_idx, margin=40, interval=10, options=[0], value=0
        )
        self.live_view_checkbox = pn.widgets.Checkbox.from_param(self.param.live_view)
        self.time_slider_widget.disabled = self.param.live_view.rx.pipe(bool)
        controls = pn.Row(
            self.time_slider_widget,
            self.live_view_checkbox,
        )
        plots = self.get_dashboard()
        return pn.Column(controls, plots)

    def get_dashboard(self):
        raise NotImplementedError(
            "a plotter class needs to implement a `get_dashboard` method"
        )

    @param.depends("live_view", watch=True)
    def _on_live_view_toggle(self):
        """Handle freezing/unfreezing data when toggling live view."""
        if self.live_view:
            self._frozen_state = None
        else:
            self._frozen_state = self.state

    @param.depends("state.data", watch=True)
    def _update_slider_for_live_data(self):
        # TODO: have single dataset with single time array for all ids, what if two
        # IDSs have different times?
        state_data = next(iter(self.state.data.values()), None)
        if not state_data:
            return
        num_steps = len(state_data.time)
        self.time_slider_widget.options = list(range(num_steps))
        if self.live_view:
            self.active_state = self.state
            self.time_idx = num_steps - 1
        else:
            self.active_state = self._frozen_state

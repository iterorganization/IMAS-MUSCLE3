import panel as pn
import param


class BaseState(param.Parameterized):
    data = param.Dict(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_and_trigger(self, ids):
        self.extract(ids)
        self.param.trigger("data")

    def extract(self, ids):
        raise NotImplementedError(
            "a state class needs to implement an `extract` method"
        )


class BasePlotter(param.Parameterized):
    state = param.Parameter(doc="The live state object from the simulation.")
    time_slider = param.Integer(default=0, bounds=(0, 0), label="Time Step")
    live_view = param.Boolean(default=True, label="Live View")

    def __init__(self, state, **params):
        super().__init__(state=state, **params)
        self._frozen_state = None
        self.active_state = self.state
        self.time_idx = -1

    def get_dashboard(self):
        time_slider_widget = pn.widgets.IntSlider.from_param(
            self.param.time_slider,
        )
        controls = pn.Row(
            self.param.live_view, time_slider_widget, sizing_mode="stretch_width"
        )
        plots = self.get_plots()
        return pn.Column(controls, plots)

    def get_plots(self):
        raise NotImplementedError(
            "a plotter class needs to implement a `get_plots` method"
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
        # TODO: have single dataset with single time array for all ids?
        state_data = self.state.data.get("equilibrium")
        if not state_data:
            return
        num_steps = len(state_data.time)
        bounds = (0, max(0, num_steps - 1))
        self.param.time_slider.bounds = bounds
        if self.live_view:
            self.time_slider = bounds[1]
            self.active_state = self.state
            self.time_idx = -1
        else:
            self.active_state = self._frozen_state
            self.time_idx = self.time_slider

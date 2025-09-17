import panel as pn
import param


class BaseState(param.Parameterized):
    data = param.Dict(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, ids):
        raise NotImplementedError(
            "a state class needs to implement an `extract` method"
        )


class BasePlotter(param.Parameterized):
    state = param.Parameter(doc="The live state object from the simulation.")
    time_idx = param.Integer(default=0, label="Time Step")
    live_view = param.Boolean(default=True, label="Live View")

    def __init__(self, state, **params):
        super().__init__(state=state, **params)
        self._frozen_state = None
        self.active_state = self.state

    def get_dashboard(self):
        self.time_slider_widget = pn.widgets.EditableIntSlider.from_param(
            self.param.time_idx
        )
        self.live_view_checkbox = pn.widgets.Checkbox.from_param(self.param.live_view)
        self.time_slider_widget.disabled = self.param.live_view.rx.pipe(bool)
        controls = pn.Row(
            self.time_slider_widget,
            self.live_view_checkbox,
            sizing_mode="stretch_width",
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
        # TODO: have single dataset with single time array for all ids, what if two
        # IDSs have different times?
        state_data = next(iter(self.state.data.values()), None)
        if not state_data:
            return
        num_steps = len(state_data.time)
        bounds = (0, max(0, num_steps - 1))
        self.param.time_idx.bounds = bounds
        if self.live_view:
            self.active_state = self.state
            self.time_idx = bounds[1]
        else:
            self.active_state = self._frozen_state

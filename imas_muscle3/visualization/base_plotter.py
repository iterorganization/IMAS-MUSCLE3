import functools
import logging
import random

import holoviews as hv
import numpy as np
import panel as pn
import param
import xarray as xr
from panel.viewable import Viewable, Viewer

from imas_muscle3.visualization.base_state import BaseState, Dim, Variable
from imas_muscle3.visualization.resizable_float_panel import (
    ResizableFloatPanel,
)

logger = logging.getLogger()


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
        self.time_label = pn.pane.Markdown(  # type: ignore[no-untyped-call]
            visible=self.param._live_view.rx.bool()
        )
        controls = pn.Row(
            self.live_view_checkbox, self.time_slider_widget, self.time_label
        )
        plots = self.get_dashboard()
        self.float_panels = pn.Column(sizing_mode="stretch_width")

        ui = pn.Row()
        if self._state.auto:
            self.variable_selector = pn.widgets.Select(  # type: ignore
                width=400
            )
            add_plot_button = pn.widgets.Button(  # type: ignore
                name="Add Plot",
                button_type="primary",
                on_click=self._add_plot_callback,
            )
            close_all_button = pn.widgets.Button(  # type: ignore
                name="Close All Plots",
                button_type="danger",
                on_click=self._close_all_plots_callback,
            )

            self.plotting_controls = pn.Row(
                self.variable_selector,
                add_plot_button,
                close_all_button,
                sizing_mode="stretch_width",
                align="center",
            )

            self.filter_input = pn.widgets.TextInput(  # type: ignore
                placeholder="Filter...",
                width=400,
            )
            self.filter_input.param.watch(
                self._update_filter_view, "value_input"
            )
            automatic_ui = pn.Column(
                self.filter_input,
                self.variable_selector,
                pn.Row(add_plot_button, close_all_button),
            )
            ui.append(automatic_ui)
        ui.append(controls)
        self._panel = pn.Column(ui, plots, self.float_panels)

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
        self.time_label.object = f"## t = {self.time:.5e} s"

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

    def _update_filter_view(self, event: param.Event) -> None:
        """Updates the variable selector based on the filter text."""
        filter_text = self.filter_input.value_input.lower()
        options = [
            full_path
            for full_path in self._state.variables
            if not filter_text or filter_text in full_path.lower()
        ]
        self.variable_selector.options = sorted(options)

    @param.depends("_state.variables", watch=True)  # type: ignore[misc]
    def _update_variable_selector(self) -> None:
        """Updates the variable selector when new variables are discovered."""
        self.variable_selector.options = sorted(
            list(self._state.variables.keys())
        )

    def _close_all_plots_callback(self, event: param.Event) -> None:
        """Closes all active plot panels."""
        for float_panel in self.float_panels:
            float_panel.status = "closed"

    def _add_plot_callback(self, event: param.Event) -> None:
        """Adds a new plot panel for the selected variable."""
        full_path = self.variable_selector.value
        if not full_path or full_path not in self._state.variables:
            return

        var = self._state.variables[full_path]
        if var.is_visualized:
            return  # Plot already exists

        var.is_visualized = True

        plot_func = functools.partial(
            self._plot_variable_vs_time, full_path=full_path
        )
        dynamic_plot = pn.pane.HoloViews(  # type: ignore[no-untyped-call]
            hv.DynamicMap(param.bind(plot_func, time=self.param.time)).opts(
                framewise=True, axiswise=True
            ),
            sizing_mode="stretch_both",
        )
        float_panel = ResizableFloatPanel(
            dynamic_plot,
            name=var.full_path,
            position="left-top",
            offsetx=random.randint(0, 2000),
            offsety=random.randint(0, 1000),
            contained=False,
        )

        def on_status_change(event: param.Event) -> None:
            if event.new == "closed":
                self._floatpanel_closed_callback(full_path)

        float_panel.param.watch(on_status_change, "status")
        self.float_panels.append(float_panel)

    def _floatpanel_closed_callback(
        self, full_path: str, event: param.Event = None
    ) -> None:
        """Handles cleanup when a plot panel is closed."""
        if full_path in self._state.variables:
            var = self._state.variables[full_path]
            var.is_visualized = False
            self._state.data.pop(var.full_path, None)

    def plot_empty(self, name: str, var_dim: Dim) -> hv.Element:
        """Returns an empty plot to show when no data is available."""
        title = f"No data for t = {self.time}"
        if var_dim == Dim.TWO_D:
            return hv.QuadMesh(
                (np.array([0]), np.array([0]), np.zeros((1, 1))),
                kdims=["x", "y"],
                vdims=[name],
            ).opts(title=title, responsive=True)
        return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
            title=title, responsive=True
        )

    def plot_1d(
        self, ds: xr.Dataset, var: Variable, time_index: int
    ) -> hv.Element:
        """Generates a 1D plot for a given time index."""
        data_var = ds[var.full_path].isel(time=time_index).values
        coord_name = var.coord_names[0]
        coord_var = (
            ds[f"{var.full_path}_{coord_name}"].isel(time=time_index).values
        )
        title = f"{var.full_path} (t={float(ds.time.values[time_index]):.3f}s)"
        return hv.Curve(
            (coord_var, data_var), kdims=[coord_name], vdims=[var.full_path]
        ).opts(title=title, responsive=True)

    def plot_2d(
        self, ds: xr.Dataset, var: Variable, time_index: int
    ) -> hv.Element:
        """Generates a 2D plot for a given time index."""
        y_name, x_name = var.coord_names
        data_var = ds[var.full_path].isel(time=time_index).values
        x = ds[f"{var.full_path}_{x_name}"].isel(time=time_index).values
        y = ds[f"{var.full_path}_{y_name}"].isel(time=time_index).values
        title = f"{var.full_path} (t={float(ds.time.values[time_index]):.3f}s)"

        return hv.QuadMesh(
            (x, y, data_var),
            kdims=[x_name, y_name],
            vdims=[var.full_path],
        ).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            title=title,
            responsive=True,
            xlabel=x_name,
            ylabel=y_name,
        )

    def _plot_variable_vs_time(
        self, full_path: str, time: float
    ) -> hv.Element:
        """Core plotting function that dispatches to specific plot types."""
        var = self.active_state.variables.get(full_path)
        if not var:
            return self.plot_empty("unknown", Dim.ZERO_D)

        ds = self.active_state.data.get(var.full_path)
        if ds is None or len(ds.time) == 0:
            return self.plot_empty(var.full_path, var.dimension)

        time_array = ds.time.values
        if time not in time_array:
            return self.plot_empty(var.full_path, var.dimension)

        time_index = np.where(time_array == time)[0][0]

        if var.dimension == Dim.ZERO_D:
            t_vals = time_array[: time_index + 1]
            v_vals = (
                ds[var.full_path].isel(time=slice(0, time_index + 1)).values
            )
            return hv.Curve(
                (t_vals, v_vals), kdims=["time"], vdims=[var.full_path]
            ).opts(title=f"{var.full_path} vs time", responsive=True)
        elif var.dimension == Dim.ONE_D:
            return self.plot_1d(ds, var, time_index)
        elif var.dimension == Dim.TWO_D:
            return self.plot_2d(ds, var, time_index)

    def __panel__(self) -> Viewable:
        return self._panel

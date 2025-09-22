import logging
import runpy
import webbrowser

import panel as pn
import param

from imas_muscle3.visualization.base import BasePlotter, BaseState

logger = logging.getLogger()


class VisualizationActor(param.Parameterized):
    """A visualization actor for MUSCLE3 that can visualize live data."""

    state = param.Parameter()

    def __init__(self, plot_file_path, port, md_dict):
        super().__init__()
        self.port = port
        self.server = None

        run_path = runpy.run_path(plot_file_path)
        StateClass = run_path.get("State")
        PlotterClass = run_path.get("Plotter")
        if not StateClass or not PlotterClass:
            raise NameError(
                f"{plot_file_path} must define a 'State' class and a 'Plotter' class."
            )
        if not issubclass(StateClass, BaseState):
            raise TypeError(f"'State' in {plot_file_path} must inherit from BaseState")
        if not issubclass(PlotterClass, BasePlotter):
            raise TypeError(
                f"'Plotter' in {plot_file_path} must inherit from BasePlotter"
            )

        self.state = StateClass(md_dict)
        self.plotter = PlotterClass(state=self.state)

        stop_button = pn.widgets.Button(
            name="Stop Server",
            button_type="danger",
            on_click=lambda event: self.stop_server(),
        )
        self.message_pane = pn.pane.Markdown(
            "### Waiting for data", sizing_mode="stretch_width"
        )
        self.dynamic_panel = pn.Column(
            pn.Row(stop_button, self.message_pane), self.plotter
        )
        self.start_server()

    def start_server(self):
        self.server = pn.serve(
            self.dynamic_panel,
            port=self.port,
            show=False,
            threaded=True,
            start=True,
        )
        self._open_browser()

    def stop_server(self):
        if self.server:
            self.server.stop()
            logger.info("Panel server stopped.")

    def update_time(self, time):
        self.message_pane.object = f"### Received t = {time:.3f}"

    def notify_done(self):
        self.message_pane.object = "### All data received."
        self.plotter.live_view_checkbox.visible = False
        self.plotter.time_slider_widget.disabled = False

    def _open_browser(self):
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
        logger.info(f"Dashboard is available at {url}")

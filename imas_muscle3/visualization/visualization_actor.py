import logging
import runpy
import webbrowser
from typing import Dict

import panel as pn
import param
from imas.ids_toplevel import IDSTopLevel

from imas_muscle3.visualization.base import BasePlotter, BaseState

logger = logging.getLogger()


class VisualizationActor(param.Parameterized):
    """A visualization actor for MUSCLE3 that can visualize live data."""

    state = param.Parameter()

    def __init__(
        self,
        plot_file_path: str,
        port: int,
        md_dict: Dict[str, IDSTopLevel],
        open_browser_on_start: bool,
    ):
        """Initialize the visualization actor.

        Loads a State and Plotter class from the given file path, sets up the
        Panel layout, and starts the server.
        """
        super().__init__()
        self.port = port
        self.server = None
        self.open_browser_on_start = open_browser_on_start

        run_path = runpy.run_path(plot_file_path)
        StateClass = run_path.get("State")
        PlotterClass = run_path.get("Plotter")
        if not StateClass or not PlotterClass:
            raise NameError(
                f"{plot_file_path} must define a 'State' class and "
                "a 'Plotter' class."
            )
        if not issubclass(StateClass, BaseState):
            raise TypeError(
                f"'State' in {plot_file_path} must inherit from BaseState"
            )
        if not issubclass(PlotterClass, BasePlotter):
            raise TypeError(
                f"'Plotter' in {plot_file_path} must inherit from BasePlotter"
            )

        self.state = StateClass(md_dict)
        self.plotter = PlotterClass(self.state)

        stop_button = pn.widgets.Button(  # type: ignore[no-untyped-call]
            name="Stop Server",
            button_type="danger",
            on_click=lambda event: self.stop_server(),
        )
        self.message_pane = pn.pane.Markdown(  # type: ignore[no-untyped-call]
            "### Waiting for data", sizing_mode="stretch_width"
        )
        self.dynamic_panel = pn.Column(
            pn.Row(stop_button, self.message_pane), self.plotter
        )
        self._start_server()

    def stop_server(self) -> None:
        """Stop the Panel server if running."""
        if self.server:
            self.server.stop()
            logger.info("Panel server stopped.")

    def update_time(self, time: float) -> None:
        """Update the display with the latest received simulation time."""
        self.message_pane.object = f"### Received t = {time:.5e}"

    def notify_done(self) -> None:
        """Update the display to indicate all data has been received."""
        self.message_pane.object = "### All data received."
        self.plotter.live_view_checkbox.visible = False
        self.plotter.time_slider_widget.visible = True

    def _start_server(self) -> None:
        """Start the Panel server for visualization."""
        self.server = pn.serve(  # type: ignore[no-untyped-call]
            self.dynamic_panel,
            port=self.port,
            show=False,
            threaded=True,
            start=True,
        )
        if self.open_browser_on_start:
            self._open_browser()

    def _open_browser(self) -> None:
        """Open the dashboard in the system web browser."""
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
        logger.info(f"Dashboard is available at {url}")

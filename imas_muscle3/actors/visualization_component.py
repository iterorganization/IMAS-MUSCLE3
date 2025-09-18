r"""
MUSCLE3 actor for visualization
"""

import logging
import runpy
import time
import webbrowser

import holoviews as hv
import panel as pn
import param
from imas import IDSFactory
from libmuscle import Instance
from ymmsl import Operator

from imas_muscle3.utils import get_port_list, get_setting_optional

logger = logging.getLogger()


pn.extension()
hv.extension("bokeh")


class VisualizationActor(param.Parameterized):
    """A generic Panel visualization actor for MUSCLE3."""

    state = param.Parameter()

    def __init__(self, plot_file_path, port, md_dict):
        super().__init__()
        self.port = port
        self.server = None

        ns = runpy.run_path(plot_file_path)

        StateClass = ns.get("State")
        PlotterClass = ns.get("Plotter")

        if not all((StateClass, PlotterClass)):
            raise NameError(
                f"{plot_file_path} must define a 'State' class, a 'Plotter' class."
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
            address="0.0.0.0",
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


def handle_machine_description(instance):
    ports_in = get_port_list(instance, Operator.F_INIT)
    md_dict = {}
    for port_name in ports_in:
        msg = instance.receive(port_name)
        ids_name = port_name.replace("_md_in", "")

        ids = IDSFactory().new(ids_name)
        ids.deserialize(msg.data)
        md_dict[ids_name] = ids
    return md_dict


def main() -> None:
    """MUSCLE3 execution loop."""
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()],
            Operator.F_INIT: [
                f"{ids_name}_md_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )

    visualization_actor = None
    first_run = True
    last_trigger_time = 0
    while instance.reuse_instance():
        if first_run:
            plot_file_path = instance.get_setting("plot_file_path", "str")
            port = get_setting_optional(instance, "port", 5006)
            # FIXME: there is an issue when the plotting takes much longer than it
            # takes for data to arrive from the MUSCLE actor. As a remedy, set a
            # plotting throttle interval.
            throttle_interval = get_setting_optional(instance, "throttle_interval", 0)
            keep_alive = get_setting_optional(instance, "keep_alive", False)
            md_dict = handle_machine_description(instance)
            visualization_actor = VisualizationActor(plot_file_path, port, md_dict)
            first_run = False

        is_running = True
        ports_in = get_port_list(instance, Operator.S)

        while is_running:
            for port_name in ports_in:
                msg = instance.receive(port_name)
                ids_name = port_name.replace("_in", "")

                temp_ids = IDSFactory().new(ids_name)
                temp_ids.deserialize(msg.data)
                visualization_actor.state.extract(temp_ids)
                if msg.next_timestamp is None:
                    is_running = False
            current_time = time.time()
            if current_time - last_trigger_time >= throttle_interval:
                visualization_actor.state.param.trigger("data")
                last_trigger_time = current_time
            visualization_actor.update_time(temp_ids.time[-1])

        if keep_alive:
            visualization_actor.notify_done()
        else:
            visualization_actor.stop_server()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

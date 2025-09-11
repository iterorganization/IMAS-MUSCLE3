r"""
MUSCLE3 actor for visualization
"""

import logging
import runpy
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

    states = param.Dict(default={})

    def __init__(self, plot_file_path: str, port: int):
        super().__init__()
        self.port = port
        self.server = None

        ns = runpy.run_path(plot_file_path)
        state_definitions = ns.get("STATE_DEFINITIONS")
        dashboard_layout = ns.get("DASHBOARD_LAYOUT")

        if not state_definitions or not dashboard_layout:
            raise ValueError(
                f"{plot_file_path} must define 'STATE_DEFINITIONS' and 'DASHBOARD_LAYOUT'"
            )

        self.states = {
            name: state_class() for name, state_class in state_definitions.items()
        }

        dmaps = []
        for item in dashboard_layout:
            PlotClass = item["plot_class"]
            state_name = item["state_name"]
            plot_method_name = item["plot_method"]

            if state_name not in self.states:
                raise ValueError(f"State '{state_name}' not in STATE_DEFINITIONS")

            target_state = self.states[state_name]

            plotter_instance = PlotClass(state=target_state)
            plot_callable = getattr(plotter_instance, plot_method_name)
            dmaps.append(hv.DynamicMap(plot_callable))

        self.dynamic_panel = pn.Column(*dmaps)
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

    def _open_browser(self):
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
        logger.info(f"Dashboard is available at {url}")


def main() -> None:
    """MUSCLE3 execution loop."""
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()],
        }
    )

    visualization_actor = None
    first_run = True
    while instance.reuse_instance():
        if first_run:
            plot_file_path = instance.get_setting("plot_file_path", "str")
            port = get_setting_optional(instance, "port", 5006)
            visualization_actor = VisualizationActor(plot_file_path, port)
            first_run = False

        # In this simplified model, we receive from all ports until one signals an end
        is_running = True
        ports_in = get_port_list(instance, Operator.S)

        while is_running:
            for port_name in ports_in:
                msg = instance.receive(port_name)
                ids_name = port_name.replace("_in", "")

                if ids_name in visualization_actor.states:
                    temp_ids = IDSFactory().new(ids_name)
                    temp_ids.deserialize(msg.data)

                    state_obj = visualization_actor.states[ids_name]
                    state_obj.update(temp_ids)

                if msg.next_timestamp is None:
                    is_running = False

    if visualization_actor:
        visualization_actor.stop_server()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

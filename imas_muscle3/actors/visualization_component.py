r"""
MUSCLE3 actor for visualization
"""

import logging
import time
from typing import Dict

import holoviews as hv
import panel as pn
from imas import IDSFactory
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance
from ymmsl import Operator

from imas_muscle3.utils import get_port_list, get_setting_default
from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger()


pn.extension()
hv.extension("bokeh")


def handle_machine_description(instance: Instance) -> Dict[str, IDSToplevel]:
    """Receive and deserialize all machine description IDSs.

    Returns:
        Mapping of IDS names to machine description IDSs.
    """
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
            Operator.S: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
            Operator.F_INIT: [
                f"{ids_name}_md_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )

    visualization_actor = None
    first_run = True
    last_trigger_time = 0.0
    while instance.reuse_instance():
        if first_run:
            plot_file_path = instance.get_setting("plot_file_path", "str")
            port = get_setting_default(instance, "port", 5006)
            # FIXME: there is an issue when the plotting takes much longer
            # than it takes for data to arrive from the MUSCLE actor. As a
            # remedy, set a plotting throttle interval.
            throttle_interval = get_setting_default(
                instance, "throttle_interval", 0.1
            )
            keep_alive = get_setting_default(instance, "keep_alive", False)
            open_browser = get_setting_default(instance, "open_browser", True)
            md_dict = handle_machine_description(instance)
            visualization_actor = VisualizationActor(
                plot_file_path, port, md_dict, open_browser
            )
            first_run = False
        assert visualization_actor is not None

        is_running = True
        ports_in = get_port_list(instance, Operator.S)

        while is_running:
            common_time = None
            for port_name in ports_in:
                msg = instance.receive(port_name)
                ids_name = port_name.replace("_in", "")

                temp_ids = IDSFactory().new(ids_name)
                temp_ids.deserialize(msg.data)

                # Ensure the IDSs have the same time basis
                if common_time is None:
                    common_time = temp_ids.time
                else:
                    if not (temp_ids.time == common_time).all():
                        raise ValueError(
                            f"Time mismatch detected in IDS {ids_name}"
                        )

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

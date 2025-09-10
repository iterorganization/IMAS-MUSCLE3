r"""
MUSCLE3 actor for visualization
"""

import logging
from typing import Dict

from imas import DBEntry, IDSFactory
from libmuscle import Instance
from ymmsl import Operator

from imas_muscle3.data_sink_source import sanity_check_ports
from imas_muscle3.panel_visualization import VisualizationActor
from imas_muscle3.utils import get_port_list, get_setting_optional

logger = logging.getLogger()


def main() -> None:
    """Create instance and enter submodel execution loop"""
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()],
        }
    )
    port_list_in = get_port_list(instance, Operator.S)
    sanity_check_ports(instance)
    first_run = True
    while instance.reuse_instance():
        ids_next: Dict[str, bool] = {
            port.replace("_in", ""): True for port in port_list_in
        }
        if first_run:
            plot_file_path = instance.get_setting("plot_file_path")
            port = get_setting_optional(instance, "port", 5006)
            visualization_actor = VisualizationActor(plot_file_path, port)
            first_run = False

        with DBEntry("imas:memory?path=/", "w") as db:
            while any(ids_next.values()):
                for port_name in port_list_in:
                    ids_name = port_name.replace("_in", "")
                    if ids_next.get(ids_name, True):
                        msg_in = instance.receive(port_name)
                        ids_slice = db.factory.new(ids_name)
                        ids_slice.deserialize(msg_in.data)
                        db.put_slice(ids_slice)
                        ids = db.get(ids_name)
                        visualization_actor.ids_dict[ids_name] = ids

                        ids_next[ids_name] = msg_in.next_timestamp is not None
                visualization_actor.param.trigger("ids_dict")

    visualization_actor.stop_server()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

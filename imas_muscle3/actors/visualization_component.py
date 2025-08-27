r"""
MUSCLE3 actor for visualization
"""

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from imas import DBEntry, IDSFactory
from libmuscle import Instance
from ymmsl import Operator

from imas_muscle3.utils import get_port_list, get_setting_optional
from imas_muscle3.panel_visualization.py import VisualizationActor

logger = logging.getLogger()


def main() -> None:
    """Create instance and enter submodel execution loop"""
    logger.info("Starting OLC Actor")
    instance = Instance(
        {
            Operator.S: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )

    first_run = True
    pn_viz = None
    # enter re-use loop
    while instance.reuse_instance():
        if first_run:
            plot_func = get_setting_optional(
                instance,
                "plot_func",
                default=None,
            )
            pn_viz = VisualizationActor(plot_func)
            pn_viz.start_server
            first_run = False
        port_list_in = get_port_list(instance, Operator.F_INIT)
        ids_data = {}
        for port_name in port_list_in:
            ids_name = port_name.replace("_in", "")
            msg_in = instance.receive(port_name)
            t_cur = msg_in.timestamp
            ids_data[ids_name] = getattr(IDSFactory(), ids_name)()
            ids_data[ids_name].deserialize(msg_in.data)

            #temp
            assert len(ids_data) == 1

            pn_viz.update_plot(ids_data[ids_name])

    if pn_viz is not None:
        pn.viz.server.stop()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
r"""
MUSCLE3 actor for visualization
"""

import logging
from typing import List, Optional, Tuple

import imas
from imas import DBEntry, IDSFactory
from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT
from libmuscle import Instance
from ymmsl import Operator

from imas_muscle3.data_sink_source import sanity_check_ports
from imas_muscle3.panel_visualization import VisualizationActor
from imas_muscle3.utils import get_port_list, get_setting_optional

logger = logging.getLogger()

"""
Example YMMSL:

ymmsl_version: v0.1
model:
  name: test_model
  components:
    source_component:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    visualization_component:
      implementation: visualization_component
      ports:
        f_init: [equilibrium_in]
  conduits:
    source_component.equilibrium_out: visualization_component.equilibrium_in
settings:
  source_component.source_uri: imas:hdf5?path=/home/sebbe/projects/iter_python/IMAS-MUSCLE3/data/nice-input-dd4
implementations:
  visualization_component:
    executable: python
    args: -u -m imas_muscle3.actors.visualization_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  source_component:
    threads: 1
  visualization_component:
    threads: 1
"""


def main() -> None:
    """Create instance and enter submodel execution loop"""
    instance = Instance(
        {
            Operator.F_INIT: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )
    visualization_actor = VisualizationActor(None)
    visualization_actor.start_server()
    first_run = True
    while instance.reuse_instance():
        if first_run:
            port_list_in = get_port_list(instance, Operator.F_INIT)
            sanity_check_ports(instance)
            first_run = False

        # F_INIT
        handle_visualization(instance, port_list_in, visualization_actor)
    visualization_actor.stop_server()


def handle_visualization(
    instance: Instance, port_list: List[str], visualization_actor
) -> Tuple[float, Optional[float]]:
    """Loop through sink ids_names and receive all incoming messages"""
    t_cur = 0.0
    t_next = None
    for port_name in port_list:
        ids_name = port_name.replace("_in", "")
        msg_in = instance.receive(port_name)
        t_cur = msg_in.timestamp
        t_next = msg_in.next_timestamp
        ids_data = getattr(IDSFactory(), ids_name)()
        ids_data.deserialize(msg_in.data)
        visualization_actor.eq = ids_data
        visualization_actor.param.trigger("eq")
    return t_cur, t_next


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

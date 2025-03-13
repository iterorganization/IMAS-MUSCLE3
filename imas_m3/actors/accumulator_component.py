"""
MUSCLE3 actor performing IDS timeslice accumulation.

Only compatible with actors that send out whether their next timestamp in None.

Receives multiple timeslices for a given IDS on the S port and sends out the
IDS with all accumulated timeslices on the O_F port.
"""

import logging
from typing import Dict, List

from imaspy import DBEntry, IDSFactory
from libmuscle import Instance, Message
from ymmsl import Operator

from imas_m3.utils import get_port_list

logger = logging.getLogger()


def main() -> None:
    """Create instance and enter submodel execution loop"""
    logger.info("Starting accumulator Actor")
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()]
            + [f"{ids_name}_t_next" for ids_name in IDSFactory().ids_names()],
            Operator.O_F: [f"{ids_name}_out" for ids_name in IDSFactory().ids_names()],
        }
    )
    port_list_in = get_port_list(instance, Operator.S)
    port_list_out = get_port_list(instance, Operator.O_F)
    sanity_check_ports(port_list_in, port_list_out)

    while instance.reuse_instance():

        ids_next: Dict[str, bool] = {}
        first_round = True

        with DBEntry("imas:memory?path=/", "w") as db:
            while first_round or any(ids_next.values()):
                first_round = False
                for port_name in port_list_in:
                    ids_name = port_name.replace("_in", "")
                    if ids_next.get(ids_name, True) and not port_name.endswith(
                        "_t_next"
                    ):
                        # receive _in
                        msg_in = instance.receive(port_name)
                        ids = getattr(IDSFactory(), ids_name)()
                        ids.deserialize(msg_in.data)
                        db.put_slice(ids)

                        # receive _t_next
                        msg_in = instance.receive(port_name.replace("_in", "_t_next"))
                        ids_next[ids_name] = msg_in.next_timestamp is not None

            # send output
            for port_name in port_list_out:
                ids_name = port_name.replace("_out", "")
                ids = db.get(ids_name)
                msg_out = Message(ids.time[0], data=ids.serialize())
                instance.send(port_name, msg_out)


def sanity_check_ports(port_list_in: List[str], port_list_out: List[str]) -> None:
    """Check whether any obvious problems are present in the instance config"""
    # check port name
    for port_name in port_list_in:
        if not port_name.endswith(("_in", "_t_next")):
            raise Exception(
                "Incoming port names should use the format '*ids_name*_in' or"
                "'*ids_name*_t_next'."
                f"Problem port is {port_name}."
            )
    for port_name in port_list_out:
        if not port_name.endswith("_out"):
            raise Exception(
                "Outgoing port names should use the format '*ids_name*_out'. "
                f"Problem port is {port_name}."
            )
    # check matching incoming and outgoing ports
    ids_in_set = set(
        [port.replace("_in", "") for port in port_list_in if port.endswith("_in")]
    )
    ids_out_set = set([port.replace("_out", "") for port in port_list_out])
    if len(ids_in_set & ids_out_set) > 0:
        raise Exception(
            "Any incoming port should have an outgoing port and vice versa."
            f"Problem ports are {ids_in_set & ids_out_set}."
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

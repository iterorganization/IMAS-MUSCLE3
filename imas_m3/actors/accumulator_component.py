"""
MUSCLE3 actor performing IDS timeslice accumulation.

This actor can receive timeslices for various IDSs at the same time on the S port,
keeping track of whether or not it was the last timeslice for a given inner loop.
Optionally the 't_next' S port is used to override this behavior and match with the
last timeslice of a specific actor. It then sends out all the IDSs with all accumulated
timeslices on the O_F port.

If no actors are available that pass information for the next_timestep, it will default
to None and this actor will only be able to receive the first timeslice.

This actor might have difficulty handling other actors with dynamic timesteps that
cannot accurately predict whether their current timeslice will be the last, possibly
leading to deadlocks. It is advised to use predictable or constant timestepping.
"""

import logging
from typing import Dict, List

from imas import DBEntry, IDSFactory
from libmuscle import Instance, Message
from ymmsl import Operator

from imas_m3.utils import get_port_list

logger = logging.getLogger()


def main() -> None:
    """Create instance and accumulate IDS timeslices before sending out the full IDS"""
    logger.info("Starting accumulator actor")
    # Initiate MUSCLE3 instance
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()]
            + ["t_next"],
            Operator.O_F: [f"{ids_name}_out" for ids_name in IDSFactory().ids_names()],
        }
    )
    # fix connected ports
    port_list_in = get_port_list(instance, Operator.S)
    if "t_next" in port_list_in:
        port_list_in.remove("t_next")
    port_list_out = get_port_list(instance, Operator.O_F)
    sanity_check_ports(port_list_in, port_list_out)

    while instance.reuse_instance():
        # keep track of whether or not each IDS should keep receiving
        ids_next: Dict[str, bool] = {
            port.replace("_in", ""): True for port in port_list_in
        }

        # use memory backend db entry to accumulate timeslices in a single IDS
        with DBEntry("imas:memory?path=/", "w") as db:
            while any(ids_next.values()):
                # loop over IDSs and receive until the last timeslice in encountered
                for port_name in port_list_in:
                    ids_name = port_name.replace("_in", "")
                    if ids_next.get(ids_name, True):
                        # receive IDS
                        msg_in = instance.receive(port_name)
                        ids = getattr(IDSFactory(), ids_name)()
                        ids.deserialize(msg_in.data)
                        db.put_slice(ids)
                        # get t_next from received IDS message
                        ids_next[ids_name] = msg_in.next_timestamp is not None
                # override t_next with optional port
                if instance.is_connected("t_next"):
                    msg_in = instance.receive("t_next")
                    if msg_in.next_timestamp is None:
                        break

            # send output with all timeslices at once
            for port_name in port_list_out:
                ids_name = port_name.replace("_out", "")
                ids = db.get(ids_name)
                msg_out = Message(ids.time[0], data=ids.serialize())
                instance.send(port_name, msg_out)


def sanity_check_ports(port_list_in: List[str], port_list_out: List[str]) -> None:
    """Check whether any obvious problems are present in the instance config"""
    # check port names
    for port_name in port_list_in:
        if not port_name.endswith("_in"):
            raise Exception(
                "Incoming port names should use the format '*ids_name*_in'."
                f"Problem port is {port_name}."
            )
    for port_name in port_list_out:
        if not port_name.endswith("_out"):
            raise Exception(
                "Outgoing port names should use the format '*ids_name*_out'. "
                f"Problem port is {port_name}."
            )
    # check matching incoming and outgoing ports
    ids_in_set = set([port.replace("_in", "") for port in port_list_in])
    ids_out_set = set([port.replace("_out", "") for port in port_list_out])
    if len(ids_in_set ^ ids_out_set) > 0:
        raise Exception(
            "Any incoming port should have an outgoing port and vice versa."
            f"Problem ports are {ids_in_set ^ ids_out_set}."
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

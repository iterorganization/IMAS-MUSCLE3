"""
MUSCLE3 actor performing IDS timeslice accumulation.

Only compatible with actors that send out whether their next timestamp in None.

Receives multiple timeslices for a given IDS on the S port and sends out the
IDS with all accumulated timeslices on the O_F port.
"""

import logging

from imaspy import DBEntry, IDSFactory
from libmuscle import Instance, Message
from ymmsl import Operator

from imas_m3.utils import get_port_list

logger = logging.getLogger()


def main():
    """Create instance and enter submodel execution loop"""
    logger.info("Starting accumulator Actor")
    instance = Instance(
        {
            Operator.S: [f"{ids_name}_in" for ids_name in IDSFactory().ids_names()]
            + [f"{ids_name}_beep" for ids_name in IDSFactory().ids_names()],
            Operator.O_F: [f"{ids_name}_out" for ids_name in IDSFactory().ids_names()],
        }
    )

    while instance.reuse_instance():
        port_list_in = get_port_list(instance, Operator.S)
        port_list_out = get_port_list(instance, Operator.O_F)

        ids_next = {}
        first_round = True

        with DBEntry("imas:memory?path=/", "w") as db:
            while first_round or any(ids_next.values()):
                first_round = False
                for port_name in port_list_in:
                    ids_name = port_name.replace("_in", "")
                    if ids_next.get(ids_name, True) and not port_name.endswith("_beep"):
                        # receive _in
                        msg_in = instance.receive(port_name)
                        ids = getattr(IDSFactory(), ids_name)()
                        ids.deserialize(msg_in.data)
                        db.put_slice(ids)

                        # receive _beep
                        msg_in = instance.receive(port_name.replace("_in", "_beep"))
                        ids_next[ids_name] = msg_in.next_timestamp is not None

            # send output
            for port_name in port_list_out:
                ids_name = port_name.replace("_out", "")
                ids = db.get(ids_name)
                msg_out = Message(ids.time[0], data=ids.serialize())
                instance.send(port_name, msg_out)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

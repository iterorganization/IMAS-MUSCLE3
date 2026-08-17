"""
MUSCLE3 actor that forwards IDSs through unchanged.

For each IDS name, independently:
- accept any IDS on any port
- pass any F_INIT message on through O_F port
- pass any S message on through O_I port
"""

import logging

from imas import IDSFactory
from libmuscle import Instance
from ymmsl.v0_2 import Operator

logger = logging.getLogger()

IDS_NAMES = IDSFactory().ids_names()


def _validate_ports(in_f: set, in_s: set, out_f: set, out_i: set) -> None:
    """Raise if any output is connected without a matching input."""
    unmatched = (out_f | out_i) - (in_f | in_s)
    if unmatched:
        raise RuntimeError(
            "passthrough: output(s) connected without a matching input "
            f"for IDS(s): {sorted(unmatched)}"
        )


def main() -> None:
    instance = Instance(
        {
            Operator.F_INIT: [f"{ids_name}_in_f" for ids_name in IDS_NAMES],
            Operator.S: [f"{ids_name}_in_s" for ids_name in IDS_NAMES],
            Operator.O_F: [f"{ids_name}_out_f" for ids_name in IDS_NAMES],
            Operator.O_I: [f"{ids_name}_out_i" for ids_name in IDS_NAMES],
        }
    )

    in_f = {n for n in IDS_NAMES if instance.is_connected(f"{n}_in_f")}
    in_s = {n for n in IDS_NAMES if instance.is_connected(f"{n}_in_s")}
    out_f = {n for n in IDS_NAMES if instance.is_connected(f"{n}_out_f")}
    out_i = {n for n in IDS_NAMES if instance.is_connected(f"{n}_out_i")}

    _validate_ports(in_f, in_s, out_f, out_i)

    active = sorted((in_f | in_s) & (out_f | out_i))

    while instance.reuse_instance():
        logger.info("passthrough: forwarding %s", active)

        for name in active:
            f_init_msg = None
            last = None
            if name in in_f:
                f_init_msg = last = instance.receive(f"{name}_in_f")
            if name in in_s:
                while True:
                    last = instance.receive(f"{name}_in_s")
                    if name in out_i:
                        instance.send(f"{name}_out_i", last)
                    if last.next_timestamp is None:
                        break
            if name in out_f:
                out_f_msg = f_init_msg if f_init_msg is not None else last
                if out_f_msg is not None:
                    instance.send(f"{name}_out_f", out_f_msg)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

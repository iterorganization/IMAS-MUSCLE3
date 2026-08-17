"""
MUSCLE3 actor performing IDS timeslice iteration.

The opposite of the accumulator actor: it receives a full multi-timeslice IDS
once per IDS name on the F_INIT port, disassembles it, and sends out the
individual timeslices one by one on the O_I port, chaining next_timestamp the
way a real timestepping source would.

The timeslices to send out are determined by settings:

- time_source_ids: names the connected input IDS whose time range is used
  to determine the output timeslices. Optional if exactly one IDS is
  connected on the F_INIT ports, in which case that IDS is used
  automatically; required otherwise.
- n_timeslices: optional amount of evenly spaced times spanning the full
  time range of time_source_ids. If not set, its native time array is
  used as-is.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from imas import DBEntry, IDSFactory
from libmuscle import Instance, InstanceFlags, Message
from ymmsl.v0_2 import Operator

from imas_muscle3.actors.accumulator_component import sanity_check_ports
from imas_muscle3.data_sink_source import fix_interpolation_method
from imas_muscle3.utils import (
    get_port_list,
    get_setting_optional,
    ids_from_message,
    ids_name_from_port,
)

logger = logging.getLogger()


@dataclass
class IteratorSettings:
    """Every setting the iterator actor reads, gathered in one place and
    read once on its first reuse (settings never change across reuses)."""

    time_source_ids: Optional[str]
    """Name of the connected input IDS whose time range determines the
    output timeslices. Optional if exactly one IDS is connected."""
    n_timeslices: Optional[int]
    """Amount of evenly spaced timeslices spanning the full time range of
    time_source_ids. If unset, its native time array is used as-is."""
    interpolation_method: int
    """Which IMAS interpolation method to use for get_slice, defaults to
    CLOSEST_INTERP."""

    @classmethod
    def from_instance(cls, instance: Instance) -> "IteratorSettings":
        return cls(
            time_source_ids=get_setting_optional(instance, "time_source_ids"),
            n_timeslices=get_setting_optional(instance, "n_timeslices"),
            interpolation_method=fix_interpolation_method(instance),
        )


def determine_t_array(
    db: DBEntry, port_list_in: List[str], settings: IteratorSettings
) -> List[float]:
    """Work out the times to slice the accumulated IDSs at, based on
    time_source_ids' own time range."""
    ids_names_in = [ids_name_from_port(port) for port in port_list_in]
    time_source_ids = settings.time_source_ids
    if time_source_ids is None:
        if len(ids_names_in) != 1:
            raise ValueError(
                "time_source_ids must be set when the number of connected "
                f"input IDSs is not exactly one (connected: {ids_names_in})."
            )
        time_source_ids = ids_names_in[0]
    elif time_source_ids not in ids_names_in:
        raise ValueError(
            f"time_source_ids '{time_source_ids}' is not among the "
            f"connected input IDSs: {ids_names_in}."
        )

    t_array = list(db.get(time_source_ids, lazy=True).time)
    if settings.n_timeslices is None:
        return t_array

    if not t_array:
        raise ValueError(f"'{time_source_ids}' has no valid time array.")
    t_min, t_max = min(t_array), max(t_array)
    n = settings.n_timeslices
    if n == 1:
        return [t_min]
    step = (t_max - t_min) / (n - 1)
    return [t_min + i * step for i in range(n)]


def main() -> None:
    """Create instance and disassemble a full IDS into timeslices sent out
    one by one."""
    logger.info("Starting iterator actor")
    instance = Instance(
        {
            Operator.F_INIT: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
            Operator.O_I: [
                f"{ids_name}_out" for ids_name in IDSFactory().ids_names()
            ],
        },
        flags=InstanceFlags.USES_CHECKPOINT_API,
    )
    port_list_in = get_port_list(instance, Operator.F_INIT)
    port_list_out = get_port_list(instance, Operator.O_I)
    sanity_check_ports(port_list_in, port_list_out)

    first_run = True
    while instance.reuse_instance():
        if first_run:
            settings = IteratorSettings.from_instance(instance)
            first_run = False

        t_array: List[float]
        with DBEntry("imas:memory?path=/", "w") as db:
            if instance.resuming():
                msg = instance.load_snapshot()
                t_array = msg.data[0]
                for ids_name, obj in msg.data[1].items():
                    db.put(ids_from_message(ids_name, obj))
            if instance.should_init():
                for port_name in port_list_in:
                    ids_name = ids_name_from_port(port_name)
                    msg_in = instance.receive(port_name)
                    db.put(ids_from_message(ids_name, msg_in.data))
                t_array = determine_t_array(db, port_list_in, settings)

            t_cur = 0.0
            for i, t_cur in enumerate(t_array):
                next_t = t_array[i + 1] if i + 1 < len(t_array) else None
                for port_name in port_list_out:
                    ids_name = port_name.replace("_out", "")
                    slice_out = db.get_slice(
                        ids_name, t_cur, settings.interpolation_method
                    )
                    msg_out = Message(
                        t_cur,
                        data=slice_out.serialize(),
                        next_timestamp=next_t,
                    )
                    instance.send(port_name, msg_out)
                if instance.should_save_snapshot(t_cur):
                    data = [
                        t_array[i + 1 :],
                        {
                            ids_name: db.get(ids_name).serialize()
                            for ids_name in (
                                ids_name_from_port(port)
                                for port in port_list_in
                            )
                        },
                    ]
                    instance.save_snapshot(Message(t_cur, data=data))
        if instance.should_save_final_snapshot():
            instance.save_final_snapshot(Message(t_cur))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

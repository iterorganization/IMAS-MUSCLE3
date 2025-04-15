"""
Muscled data sink and/or source actor.

- Assumes that the port names for the conduits going out and in have
    the format ``*ids_name*_in`` and ``*ids_name*_out``, will fail otherwise.
- Set sink_uri and/or source_uri in the settings to determine which DBEntry
    is used as data sink and/or source.
- You can set the occurrence number per port with the optional setting
    ``*ids_name*_out_occ``
- For now the only available ports for the components are:
    source: O_I
    sink: F_INIT
    sink_source: F_INIT, O_F
- Available settings are
    sink_uri: which db entry uri the data should be saved to 
    source_uri: which db entry uri the data should be loaded from
    t_min: left boundary of loaded time range
    t_max: right boundary of loaded time range
    interpolation_method: which imas interpolation method to use for load,
                          defaults to CLOSEST_INTERP
    dd_version: which IMAS DD version should be used
    {port_name}_occ: occurrence number for loading and saving of given ids

How to use in ymmsl file::

    model:
        name: example_model
        components:
            macro:
                implementation: source_component
                ports:
                o_i: [core_profiles_out]
            micro:
                implementation: sink_component
                ports:
                f_init: [core_profiles_in]
        conduits:
            macro.core_profiles_out: micro.core_profiles_in
    settings:
        macro.source_uri: source_uri
        micro.sink_uri: sink_uri
    implementations:
        sink_component:
            executable: python
            args: -u -m pds.utils.sink_component
        source_component:
            executable: python
            args: -u -m pds.utils.source_component
"""

import logging
from typing import List, Optional

from imas import DBEntry, IDSFactory
from imas.ids_defs import CLOSEST_INTERP, PREVIOUS_INTERP, LINEAR_INTERP
from libmuscle import Instance, Message
from ymmsl import Operator

from imas_m3.utils import get_port_list, get_setting_optional

# TODO: enable specifying time range
# TODO: setting for full ids instead of separate time_slices
# TODO: handle sanity checks for timestamps
# TODO: make interp_method a setting
# TODO: make fully flexible single component


def muscled_sink() -> None:
    """Implementation of sink component"""
    instance = Instance(
        {
            Operator.F_INIT: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )
    first_run = True
    while instance.reuse_instance():
        if first_run:
            dd_version = get_setting_optional(instance, "dd_version")
            sink_uri = instance.get_setting("sink_uri")
            sink_db_entry = DBEntry(sink_uri, "w", dd_version=dd_version)
            port_list_in = get_port_list(instance, Operator.F_INIT)
            sanity_check_ports(instance)
            first_run = False

        # F_INIT
        handle_sink(instance, sink_db_entry, port_list_in)
    sink_db_entry.close()


def muscled_source() -> None:
    """Implementation of source component"""
    instance = Instance(
        {
            Operator.O_I: [f"{ids_name}_out" for ids_name in IDSFactory().ids_names()],
        }
    )
    first_run = True
    while instance.reuse_instance():
        if first_run:
            dd_version = get_setting_optional(instance, "dd_version")
            source_uri = instance.get_setting("source_uri")
            source_db_entry = DBEntry(source_uri, "r", dd_version=dd_version)
            port_list_out = get_port_list(instance, Operator.O_I)
            t_array = source_db_entry.get(
                port_list_out[0].replace("_out", ""), lazy=True
            ).time
            t_min = max(
                get_setting_optional(instance, "t_min", default=-1e20), t_array[0]
            )
            t_max = min(
                get_setting_optional(instance, "t_max", default=1e20), t_array[-1]
            )
            t_array = [t for t in t_array if t_min <= t <= t_max]
            sanity_check_ports(instance)
            first_run = False

        for t_inner in t_array:
            # O_I
            handle_source(instance, source_db_entry, port_list_out, t_inner)
    source_db_entry.close()


def muscled_sink_source() -> None:
    """Implementation of hybrid sink source component"""
    instance = Instance(
        {
            Operator.F_INIT: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
            Operator.O_F: [f"{ids_name}_out" for ids_name in IDSFactory().ids_names()],
        }
    )
    first_run = True
    while instance.reuse_instance():
        if first_run:
            dd_version = get_setting_optional(instance, "dd_version")
            sink_uri = get_setting_optional(instance, "sink_uri")
            source_uri = instance.get_setting("source_uri")
            sink_db_entry = DBEntry(sink_uri, "w", dd_version=dd_version)
            source_db_entry = DBEntry(source_uri, "r", dd_version=dd_version)
            port_list_in = get_port_list(instance, Operator.F_INIT)
            port_list_out = get_port_list(instance, Operator.O_F)
            sanity_check_ports(instance)
            first_run = False

        # F_INIT
        if sink_uri is not None:
            t_cur = handle_sink(instance, sink_db_entry, port_list_in) or 0
        # O_F
        handle_source(instance, source_db_entry, port_list_out, t_cur)

    sink_db_entry.close()
    source_db_entry.close()


def handle_source(
    instance: Instance,
    db_entry: Optional[DBEntry],
    port_list: List[str],
    t_cur: float,
) -> None:
    """Loop through source ids_names and send all outgoing messages"""
    if db_entry is None:
        return

    for port_name in port_list:
        ids_name = port_name.replace("_out", "")
        occ = get_setting_optional(instance, f"{port_name}_occ", default=0)
        interp_method = fix_interpolation_method(instance)
        slice_out = db_entry.get_slice(
            ids_name=ids_name,
            occurrence=occ,
            time_requested=t_cur,
            interpolation_method=interp_method,
        )
        msg_out = Message(t_cur, data=slice_out.serialize())
        instance.send(port_name, msg_out)


def handle_sink(
    instance: Instance,
    db_entry: Optional[DBEntry],
    port_list: List[str],
) -> Optional[float]:
    """Loop through sink ids_names and receive all incoming messages"""
    t_cur = None
    for port_name in port_list:
        ids_name = port_name.replace("_in", "")
        occ = get_setting_optional(instance, f"{port_name}_occ", default=0)
        msg_in = instance.receive(port_name)
        t_cur = msg_in.timestamp
        if db_entry is not None:
            # ids_data = getattr(imas, ids_name)()
            ids_data = getattr(IDSFactory(), ids_name)()
            ids_data.deserialize(msg_in.data)
            db_entry.put_slice(ids_data, occurrence=occ)
    return t_cur


def sanity_check_ports(instance: Instance) -> None:
    """Check whether any obvious problems are present in the instance config"""
    # check port name
    for operator, ports in instance.list_ports().items():
        for port_name in ports:
            if operator.name in ["F_INIT", "S"] and not port_name.endswith("_in"):
                raise Exception(
                    "Incoming port names should use the format '*ids_name*_in'. "
                    f"Problem port is {port_name}."
                )
            if operator.name in ["O_I", "O_F"] and not port_name.endswith("_out"):
                raise Exception(
                    "Outgoing port names should use the format '*ids_name*_out'. "
                    f"Problem port is {port_name}."
                )
    # check whether uri is provided if component acts as source
    no_source_uri = get_setting_optional(instance, "source_uri") is None
    no_source_ports = (
        len(
            instance.list_ports().get(Operator.O_I, [])
            + instance.list_ports().get(Operator.O_F, [])
        )
        == 0
    )
    if no_source_uri != no_source_ports:
        raise Exception(
            "Component needs a DBEntry URI to act as source. "
            "Add source_uri in the ymmsl settings file."
        )


def fix_interpolation_method(instance: Instance) -> int:
    setting = get_setting_optional(instance, "interpolation_method")
    if setting == 'CLOSEST_INTERP':
        interp = CLOSEST_INTERP
    elif setting == 'PREVIOUS_INTERP':
        interp = PREVIOUS_INTERP
    elif setting == 'LINEAR_INTERP':
        interp = LINEAR_INTERP
    else:
        interp = CLOSEST_INTERP
    return interp


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    muscled_sink_source()

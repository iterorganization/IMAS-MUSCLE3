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

How to use in ymmsl file (yMMSL v0.2)::

    ymmsl_version: v0.2
    models:
        example_model:
            description: minimal source -> sink example
            components:
                macro:
                    description: data source
                    implementation: source_component
                    ports:
                        o_i: [core_profiles_out]
                micro:
                    description: data sink
                    implementation: sink_component
                    ports:
                        f_init: [core_profiles_in]
            conduits:
                macro.core_profiles_out: micro.core_profiles_in
    settings:
        macro.source_uri: source_uri
        micro.sink_uri: sink_uri
    programs:
        sink_component:
            executable: python
            args: -u -m imas_muscle3.actors.sink_component
        source_component:
            executable: python
            args: -u -m imas_muscle3.actors.source_component
"""

import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from imas import DBEntry, IDSFactory
from imas.ids_defs import (
    CLOSEST_INTERP,
    IDS_TIME_MODE_INDEPENDENT,
    LINEAR_INTERP,
    PREVIOUS_INTERP,
)
from imas_core.exception import ImasCoreBackendException
from libmuscle import Instance, InstanceFlags, Message
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import (
    get_port_list,
    get_setting_optional,
    increment_suffix,
)

INCREMENT_MAX = 100

# TODO: enable specifying time range
# TODO: setting for full ids instead of separate time_slices
# TODO: handle sanity checks for timestamps
# TODO: make interp_method a setting
# TODO: make fully flexible single component


def get_sink_db_entry(
    sink_uri: str,
    sink_mode: str = "x",
    avoid_name_collision: bool = True,
    dd_version: Optional[str] = None,
) -> DBEntry:
    """Get DBEntry object for sink. Puts incremental suffix at end if
    already exists, but only if sink_mode is 'x' and avoid_name_collission
    setting is enabled. Suffix incrementing does not work for non-path based
    uri's. Works for both HDF5 and MDSPLUS backend."""
    for i in range(INCREMENT_MAX):
        changed = False
        try:
            sink_db_entry = DBEntry(sink_uri, sink_mode, dd_version=dd_version)
        except ImasCoreBackendException as e:
            if (
                sink_mode != "x"
                or not avoid_name_collision
                or (
                    "already exists" not in str(e.message)
                    and "exists already" not in str(e.message)
                )
                or "path=" not in urlparse(sink_uri).query
            ):
                raise e
            else:
                sink_uri = increment_suffix(sink_uri)
                changed = True
        else:
            if changed:
                logging.warning(
                    "Provided sink path already exists, "
                    f"wrote to {sink_uri} instead."
                )
            return sink_db_entry
    raise ValueError(
        "Did not manage to open DBEntry. A DBEntry already exists at given "
        "path, as well as *path*_1 up to *path*_99."
    )


def muscled_sink() -> None:
    """Implementation of sink component"""
    # we can leave out port names on f_init since any connected port will
    # automatically be put there, this minimizes logs getting clogged with
    # prereceive messages
    sink_db_entry = None
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)
    first_run = True
    while instance.reuse_instance():
        if first_run:
            dd_version = get_setting_optional(instance, "dd_version")
            sink_mode = instance.get_setting("sink_mode", default="x")
            sink_uri = instance.get_setting("sink_uri")
            avoid_name_collision = instance.get_setting(
                "avoid_name_collision", default=True
            )
            sink_db_entry = get_sink_db_entry(
                sink_uri,
                sink_mode=sink_mode,
                avoid_name_collision=avoid_name_collision,
                dd_version=dd_version,
            )
            port_list_in = get_port_list(instance, Operator.F_INIT)
            sanity_check_ports(instance)
            first_run = False

        # F_INIT
        handle_sink(instance, sink_db_entry, port_list_in)
    if sink_db_entry is not None:
        sink_db_entry.close()


def muscled_source() -> None:
    """Implementation of source component"""
    source_db_entry = None
    instance = Instance(
        {
            Operator.O_I: [
                f"{ids_name}_out" for ids_name in IDSFactory().ids_names()
            ],
        },
        flags=InstanceFlags.USES_CHECKPOINT_API,
    )
    first_run = True
    while instance.reuse_instance():
        if first_run:
            iterative = instance.get_setting("iterative", default=True)
            dd_version = get_setting_optional(instance, "dd_version")
            source_uri = instance.get_setting("source_uri")
            source_db_entry = DBEntry(source_uri, "r", dd_version=dd_version)
            port_list_out = get_port_list(instance, Operator.O_I)
            t_array = time_array_from_IDS(
                source_db_entry, port_list_out, instance
            )
            sanity_check_ports(instance)
            first_run = False

        if instance.resuming():
            msg = instance.load_snapshot()
            t_cur = msg.timestamp
            t_array = [t for t in t_array if t > t_cur]
        if instance.should_init():
            pass

        if iterative:
            for i, t_inner in enumerate(t_array):
                # O_I
                if i < len(t_array) - 1:
                    next_t = t_array[i + 1]
                else:
                    next_t = None
                handle_source(
                    instance,
                    source_db_entry,
                    port_list_out,
                    t_inner,
                    next_timestamp=next_t,
                )
                if instance.should_save_snapshot(t_inner):
                    msg = Message(t_inner)
                    instance.save_snapshot(msg)
        else:
            handle_source(
                instance,
                source_db_entry,
                port_list_out,
                t_array[0],
                iterative=False,
            )

        if instance.should_save_final_snapshot():
            msg = Message(t_array[-1])
            instance.save_final_snapshot(msg)
    if source_db_entry is not None:
        source_db_entry.close()


def muscled_sink_source() -> None:
    """Implementation of hybrid sink source component"""
    sink_db_entry = None
    source_db_entry = None
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)
    sink_db_entry = None
    first_run = True
    while instance.reuse_instance():
        if first_run:
            dd_version = get_setting_optional(instance, "dd_version")
            sink_mode = instance.get_setting("sink_mode", default="x")
            sink_uri = get_setting_optional(instance, "sink_uri")
            source_uri = instance.get_setting("source_uri")
            avoid_name_collision = instance.get_setting(
                "avoid_name_collision", default=True
            )
            if isinstance(sink_uri, str):
                sink_db_entry = get_sink_db_entry(
                    sink_uri,
                    sink_mode=sink_mode,
                    avoid_name_collision=avoid_name_collision,
                    dd_version=dd_version,
                )
            source_db_entry = DBEntry(source_uri, "r", dd_version=dd_version)
            port_list_in = get_port_list(instance, Operator.F_INIT)
            port_list_out = get_port_list(instance, Operator.O_F)
            sanity_check_ports(instance)
            first_run = False

        # F_INIT
        t_cur, t_next = handle_sink(instance, sink_db_entry, port_list_in)
        # O_F
        handle_source(
            instance,
            source_db_entry,
            port_list_out,
            t_cur,
            next_timestamp=t_next,
        )

    if sink_db_entry is not None:
        sink_db_entry.close()
    if source_db_entry is not None:
        source_db_entry.close()


def handle_source(
    instance: Instance,
    db_entry: Optional[DBEntry],
    port_list: List[str],
    t_cur: float,
    next_timestamp: Optional[float] = None,
    iterative: bool = True,
) -> None:
    """Loop through source ids_names and send all outgoing messages"""
    if db_entry is None:
        return

    for port_name in port_list:
        ids_name = port_name.replace("_out", "")
        occ = instance.get_setting(f"{port_name}_occ", default=0)
        interp_method = fix_interpolation_method(instance)
        if iterative:
            slice_out = db_entry.get_slice(
                ids_name=ids_name,
                occurrence=occ,
                time_requested=t_cur,
                interpolation_method=interp_method,
            )
        else:
            slice_out = db_entry.get(
                ids_name=ids_name,
                occurrence=occ,
            )
        msg_out = Message(
            t_cur, data=slice_out.serialize(), next_timestamp=next_timestamp
        )
        instance.send(port_name, msg_out)


def handle_sink(
    instance: Instance,
    db_entry: Optional[DBEntry],
    port_list: List[str],
) -> Tuple[float, Optional[float]]:
    """Loop through sink ids_names and receive all incoming messages"""
    t_cur = 0.0
    t_next = None
    for port_name in port_list:
        ids_name = port_name.replace("_in", "")
        occ = instance.get_setting(f"{port_name}_occ", default=0)
        msg_in = instance.receive(port_name)
        t_cur = msg_in.timestamp
        t_next = msg_in.next_timestamp
        if db_entry is not None:
            ids_data = getattr(IDSFactory(), ids_name)()
            ids_data.deserialize(msg_in.data)
            if (
                len(ids_data.time) > 1
                or ids_data.ids_properties.homogeneous_time
                == IDS_TIME_MODE_INDEPENDENT
            ):
                db_entry.put(ids_data, occurrence=occ)
            else:
                db_entry.put_slice(ids_data, occurrence=occ)
    return t_cur, t_next


def sanity_check_ports(instance: Instance) -> None:
    """Check whether any obvious problems are present in the instance config"""
    # check port name
    for operator, ports in instance.list_ports().items():
        for port_name in ports:
            if operator.name in ["F_INIT", "S"] and not port_name.endswith(
                "_in"
            ):
                raise Exception(
                    "Incoming port names should use the format "
                    f"'*ids_name*_in'. Problem port is {port_name}."
                )
            if operator.name in ["O_I", "O_F"] and not port_name.endswith(
                "_out"
            ):
                raise Exception(
                    "Outgoing port names should use the format "
                    f"'*ids_name*_out'. Problem port is {port_name}."
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
    setting = instance.get_setting("interpolation_method", default="closest")
    if setting == "closest":
        interp = CLOSEST_INTERP
    elif setting == "previous":
        interp = PREVIOUS_INTERP
    elif setting == "linear":
        interp = LINEAR_INTERP
    else:
        interp = CLOSEST_INTERP
    return interp


def time_array_from_IDS(
    db_entry: DBEntry, port_list: List[str], instance: Instance
) -> List[float]:
    for port in port_list:
        t_array = db_entry.get(port.replace("_out", ""), lazy=True).time
        if len(t_array) > 0:
            t_min = instance.get_setting("t_min", default=-1e20)
            t_min = max(t_min, t_array[0])
            t_max = instance.get_setting("t_max", default=1e20)
            t_max = min(t_max, t_array[-1])
            t_array = [t for t in t_array if t_min <= t <= t_max]
            return t_array
    raise ValueError("No IDS with valid time array found.")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    muscled_sink_source()

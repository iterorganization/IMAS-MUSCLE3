"""
MUSCLE3 actor for visualization
"""

import logging
import time
from typing import Dict

import holoviews as hv
import panel as pn
from imas import IDSFactory
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, InstanceFlags, Message
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import get_port_list
from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger()


pn.extension(notifications=True)
hv.extension("bokeh")


def handle_machine_description(
    instance: Instance, first_run: bool
) -> Dict[str, IDSToplevel]:
    """Receive and deserialize all machine description IDSs.

    Returns:
        Mapping of IDS names to machine description IDSs.
    """
    md_dict = {}

    md_ports_in = [
        p for p in get_port_list(instance, Operator.S) if p.endswith("_md_in")
    ]
    for port_name in md_ports_in:
        msg = instance.receive(port_name)
        # In order for checkpointing to work, we must receive the
        # machine description messages coming in on the S-port
        if not first_run:
            continue
        ids_name = port_name.replace("_md_in", "")
        ids = IDSFactory().new(ids_name)
        ids.deserialize(msg.data)
        md_dict[ids_name] = ids
    return md_dict


def main() -> None:
    """MUSCLE3 execution loop."""
    instance = Instance(
        {
            # Optional driver trigger: when connected, the actor reuses once
            # per received message (e.g. one per outer-loop iteration) and
            # keeps the server alive across them. When unconnected it runs
            # a single pass.
            Operator.F_INIT: ["trigger_in"],
            Operator.S: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ]
            + [f"{ids_name}_md_in" for ids_name in IDSFactory().ids_names()],
        },
        flags=InstanceFlags.USES_CHECKPOINT_API,
    )

    visualization_actor = None
    first_run = True
    last_trigger_time = 0.0
    ports_in = [
        p
        for p in get_port_list(instance, Operator.S)
        if not p.endswith("_md_in")
    ]
    keep_alive = False
    while instance.reuse_instance():
        if instance.resuming():
            pass
        if instance.should_init():
            pass

        # Consume the driver trigger (if any) that gates this reuse.
        if instance.is_connected("trigger_in"):
            instance.receive("trigger_in")

        plot_file_path = instance.get_setting("plot_file_path", "str")
        # If port is not specified, use a random available port
        port = instance.get_setting("port", default=0)
        # FIXME: there is an issue when the plotting takes much longer
        # than it takes for data to arrive from the MUSCLE actor. As a
        # remedy, set a plotting throttle interval.
        throttle_interval = instance.get_setting(
            "throttle_interval", default=0.1
        )
        keep_alive = instance.get_setting("keep_alive", default=False)
        open_browser = instance.get_setting("open_browser", default=True)
        automatic_mode = instance.get_setting("automatic_mode", default=False)
        extract_all = instance.get_setting(
            "automatic_extract_all", default=False
        )

        # for mypy
        assert port is not None
        assert open_browser is not None
        assert extract_all is not None
        assert automatic_mode is not None
        assert throttle_interval is not None

        is_running = True
        while is_running:
            md_dict = handle_machine_description(instance, first_run)
            if first_run:
                visualization_actor = VisualizationActor(
                    plot_file_path,
                    port,
                    md_dict,
                    open_browser,
                    extract_all,
                    automatic_mode,
                )
                first_run = False

            assert visualization_actor is not None
            common_time = None
            for port_name in ports_in:
                msg = instance.receive(port_name)
                t_cur = msg.timestamp
                ids_name = port_name.replace("_in", "")

                temp_ids = IDSFactory().new(ids_name)
                temp_ids.deserialize(msg.data)

                # Ensure the IDSs have the same time basis
                if common_time is None:
                    common_time = temp_ids.time
                else:
                    if not (temp_ids.time == common_time).all():
                        raise ValueError(
                            f"Time mismatch detected in IDS {ids_name}"
                        )

                visualization_actor.state.extract_data(temp_ids)
                if msg.next_timestamp is None:
                    is_running = False
            current_time = time.time()
            if current_time - last_trigger_time >= throttle_interval:
                visualization_actor.state.param.trigger("data")
                last_trigger_time = current_time
            visualization_actor.update_time(temp_ids.time[-1])

            if instance.should_save_snapshot(t_cur):
                msg = Message(t_cur)
                instance.save_snapshot(msg)

        assert visualization_actor is not None
        visualization_actor.state.param.trigger("data")

        if instance.should_save_final_snapshot():
            msg = Message(t_cur)
            instance.save_final_snapshot(msg)

    # Finalize once, after the last reuse, so the server survives across
    # iterations when driven by trigger_in.
    if visualization_actor is not None:
        if keep_alive:
            visualization_actor.notify_done()
        else:
            visualization_actor.stop_server()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

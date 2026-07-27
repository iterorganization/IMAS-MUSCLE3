"""
MUSCLE3 actor for visualization
"""

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Dict

import holoviews as hv
import panel as pn
from imas import IDSFactory
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, InstanceFlags, Message
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import (
    get_port_list,
    ids_from_message,
    ids_name_from_port,
)
from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger()


pn.extension(notifications=True)
hv.extension("bokeh")


@dataclass
class VisualizationSettings:
    """Every setting the visualization actor reads, gathered in one place
    and read once per reuse."""

    plot_file_path: str
    """Path to write the plot/dashboard file to."""
    port: int
    """Port for the visualization server."""
    throttle_interval: float
    """Minimum time (s) between plot refreshes."""
    keep_alive: bool
    """Keep the visualization server running after the last reuse."""
    open_browser: bool
    """Open a browser tab pointed at the visualization server on start."""
    automatic_mode: bool
    """Run without waiting for manual plot interaction."""
    extract_all: bool
    """Extract all available data instead of a curated subset."""

    @classmethod
    def from_instance(cls, instance: Instance) -> "VisualizationSettings":
        return cls(
            plot_file_path=instance.get_setting("plot_file_path", "str"),
            port=instance.get_setting("port", "int", default=0),
            # FIXME: there is an issue when the plotting takes much longer
            # than it takes for data to arrive from the MUSCLE actor. As a
            # remedy, throttle_interval sets a plotting throttle interval.
            # Fetched untyped and coerced: ymmsl settings written as e.g.
            # `0` parse as int, but libmuscle's "float" type check rejects
            # those.
            throttle_interval=float(
                instance.get_setting("throttle_interval", default=0.1)
            ),
            keep_alive=instance.get_setting(
                "keep_alive", "bool", default=False
            ),
            open_browser=instance.get_setting(
                "open_browser", "bool", default=True
            ),
            automatic_mode=instance.get_setting(
                "automatic_mode", "bool", default=False
            ),
            extract_all=instance.get_setting(
                "automatic_extract_all", "bool", default=False
            ),
        )


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
        md_dict[ids_name] = ids_from_message(ids_name, msg.data)
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
    while instance.reuse_instance():
        if instance.resuming():
            msg = instance.load_snapshot()
        if instance.should_init():
            pass

        # Consume the driver trigger (if any) that gates this reuse.
        if instance.is_connected("trigger_in"):
            instance.receive("trigger_in")

        settings = VisualizationSettings.from_instance(instance)

        is_running = True
        try:
            with contextlib.ExitStack() as stack:
                while is_running:
                    md_dict = handle_machine_description(instance, first_run)
                    if first_run:
                        visualization_actor = VisualizationActor(
                            settings.plot_file_path,
                            settings.port,
                            md_dict,
                            settings.open_browser,
                            settings.extract_all,
                            settings.automatic_mode,
                            keep_alive=settings.keep_alive,
                        )
                        stack.enter_context(visualization_actor)
                        first_run = False

                    assert visualization_actor is not None
                    for port_name in ports_in:
                        msg = instance.receive(port_name)
                        t_cur = msg.timestamp
                        ids_name = ids_name_from_port(port_name)
                        temp_ids = ids_from_message(ids_name, msg.data)

                        visualization_actor.state.extract_data(temp_ids)
                        if msg.next_timestamp is None:
                            is_running = False
                    current_time = time.time()
                    if (
                        current_time - last_trigger_time
                        >= settings.throttle_interval
                    ):
                        visualization_actor.state.param.trigger("data")
                        last_trigger_time = current_time
                    visualization_actor.update_time(temp_ids.time[-1])

                    if instance.should_save_snapshot(t_cur):
                        msg = Message(t_cur)
                        instance.save_snapshot(msg)

                assert visualization_actor is not None
                visualization_actor.state.param.trigger("data")

        except (RuntimeError, NameError, TypeError) as e:
            logging.error(f"{type(e).__name__}: {e}")

        assert visualization_actor is not None
        visualization_actor.state.param.trigger("data")

        if instance.should_save_final_snapshot():
            msg = Message(t_cur)
            instance.save_final_snapshot(msg)

    # Finalize once, after the last reuse, so the server survives across
    # iterations when driven by trigger_in.
    if visualization_actor is not None:
        if settings.keep_alive:
            visualization_actor.notify_done()
        else:
            visualization_actor.stop_server()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

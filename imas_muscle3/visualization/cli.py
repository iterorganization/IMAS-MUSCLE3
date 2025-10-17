"""
Standalone IMAS Visualization Interface - Fixed Version
"""

import logging
import threading
import time

import click
import panel as pn
from imas import DBEntry, ids_defs

from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger(__name__)
pn.extension(notifications=True)


def feed_data(
    uri: str,
    ids_name: str,
    visualization_actor: VisualizationActor,
    throttle_interval: float,
) -> None:
    """Continuously feed data into the visualization actor from an IDS."""
    try:
        with DBEntry(uri, "r") as entry:
            ids = entry.get(ids_name, lazy=True)
            times = ids.time
            last_trigger_time = 0.0

            for t in times:
                ids = entry.get_slice(ids_name, t, ids_defs.CLOSEST_INTERP)
                visualization_actor.state.extract(ids)
                visualization_actor.update_time(ids.time[-1])

                current_time = time.time()
                if current_time - last_trigger_time >= throttle_interval:
                    visualization_actor.state.param.trigger("data")
                    last_trigger_time = current_time

                # FIXME:
                time.sleep(0.01)

            visualization_actor.state.param.trigger("data")
            visualization_actor.notify_done()
            logger.info("All IDS slices processed.")
    except Exception as e:
        logger.error(f"Error in data feeder thread: {e}", exc_info=True)


@click.command()
@click.argument("uri", type=str)
@click.argument("ids_name", type=str)
@click.argument("plot_file_path", type=click.Path(exists=True))
@click.option(
    "--port", default=5006, show_default=True, help="Port to run Panel server on."
)
@click.option(
    "--extract-all",
    default=False,
    is_flag=True,
    help="Extract all time-dependent IDS data on load.",
)
@click.option(
    "--throttle-interval",
    default=0.1,
    show_default=True,
    help="Seconds between UI updates.",
)
def main(
    uri: str,
    ids_name: str,
    plot_file_path: str,
    port: int,
    extract_all: bool,
    throttle_interval: float,
):
    """CLI to run the visualization actor as standalone application, without needing MUSCLE3.

    Example:
        python visualize_standalone.py "imas:hdf5?path=/path/to/data" equilibrium /path/to/plot_file.py
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    visualization_actor = VisualizationActor(
        plot_file_path=plot_file_path,
        md_dict={},
        port=port,
        open_browser_on_start=True,
        extract_all=extract_all,
    )

    feeder_thread = threading.Thread(
        target=feed_data,
        args=(uri, ids_name, visualization_actor, throttle_interval),
        daemon=False,
    )
    logger.info("Waiting for browser to load...")
    time.sleep(3)
    logger.info(f"Loading {ids_name} from {uri}...")
    feeder_thread.start()

    try:
        feeder_thread.join()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        visualization_actor.stop_server()


if __name__ == "__main__":
    main()

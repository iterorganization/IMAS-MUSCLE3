"""
Standalone IMAS Visualization Interface
"""

import logging
import threading
import time

import click
import imas
import panel as pn
from imas import DBEntry, ids_defs

from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger(__name__)
pn.extension(notifications=True)


def feed_data(
    entry: DBEntry,
    ids_name: str,
    visualization_actor: VisualizationActor,
    throttle_interval: float,
) -> None:
    """Continuously feed data into the visualization actor from an IDS."""
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

    visualization_actor.notify_done()
    logger.info("All IDS slices processed.")


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
    """Standalone visualization of an IMAS IDS using a given PLOT_FILE_PATH.

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

    entry = imas.DBEntry(uri, "r")

    feeder_thread = threading.Thread(
        target=feed_data,
        args=(entry, ids_name, visualization_actor, throttle_interval),
        daemon=True,
    )
    feeder_thread.start()

    try:
        while feeder_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        visualization_actor.stop_server()
        entry.close()


if __name__ == "__main__":
    main()

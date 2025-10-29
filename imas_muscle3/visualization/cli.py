"""
Standalone IMAS Visualization Interface
"""

import logging
import threading
import time
from typing import Dict, List, Tuple

import click
import imas
import panel as pn
from imas.ids_toplevel import IDSToplevel

from imas_muscle3.visualization.visualization_actor import VisualizationActor

logger = logging.getLogger(__name__)
pn.extension(notifications=True)


def feed_data(
    uri: str,
    ids_in_entry: list[str],
    visualization_actor: VisualizationActor,
    throttle_interval: float,
) -> None:
    """Continuously feed data into the visualization actor from an IDS.

    Args:
        uri: URI to load IDSs from.
        visualization_actor: The visualization actor object.
        throttle_interval: Interval how often the UI data is updated.
    """

    first_run = True
    try:
        with imas.DBEntry(uri, "r") as entry:
            last_trigger_time = 0.0
            for ids_name in ids_in_entry:
                if first_run:
                    ids = entry.get(ids_name, lazy=True)
                    times = ids.time
                    first_run = False

                for t in times:
                    ids = entry.get_slice(
                        ids_name, t, imas.ids_defs.CLOSEST_INTERP
                    )
                    visualization_actor.state.extract_data(ids)
                    visualization_actor.update_time(ids.time[-1])

                    current_time = time.time()
                    if current_time - last_trigger_time >= throttle_interval:
                        visualization_actor.state.param.trigger("data")
                        last_trigger_time = current_time

                    # FIXME: panel doesn't load if we do not add sleep
                    time.sleep(0.01)

            visualization_actor.state.param.trigger("data")
            visualization_actor.notify_done()
            logger.info("All IDS slices processed.")
    except Exception as e:
        logger.error(f"Error in data feeder thread: {e}", exc_info=True)


def get_available_ids(entry: imas.DBEntry) -> List[str]:
    """Return list of availble IDS names in an IMAS entry.

    Args:
        entry: The IMAS entry to check for available IDSs.

    Returns:
        List of available IDS names.
    """
    factory = entry.factory
    ids_names = factory.ids_names()
    ids_in_entry = []
    for ids_name in ids_names:
        try:
            entry.get(ids_name, lazy=True)
            ids_in_entry.append(ids_name)
        except imas.exception.DataEntryException:
            pass
    return ids_in_entry


def create_md_dict(
    default_entry: imas.DBEntry,
    md: Tuple[str],
    ids_in_entry: list[str],
) -> Dict[str, IDSToplevel]:
    """Convert --md args into a dictionary of IDS objects.

    Args:
        default_entry: Default dbentry to use if no --md is provided.
        md: Tuple of md cli arguments like 'ids_name=imas_uri'.
        ids_in_entry: List of IDSs in the default entry.

    Returns:
        Dictionary containing mapping from IDS name to IDS object.
    """
    md_dict = {}
    for md_arg in md:
        if "=" not in md_arg:
            raise click.BadParameter(
                f"Invalid machine description entry '{md_arg}'. "
                "Expected input to be in the format name=uri"
            )
        ids_name, value = md_arg.split("=", 1)
        md_uri = value.strip()
        with imas.DBEntry(md_uri, "r") as dbentry:
            md_ids = dbentry.get(ids_name)
        md_dict[ids_name.strip()] = md_ids

    for ids_name in ids_in_entry:
        if ids_name not in md_dict:
            md_ids = default_entry.get(ids_name, lazy=True)
            md_dict[ids_name] = md_ids
    return md_dict


@click.command()
@click.argument("uri", type=str)
@click.argument("plot_file_path", type=click.Path(exists=True))
@click.option(
    "--port",
    default=5006,
    show_default=True,
    help="Port to run Panel server on.",
)
@click.option(
    "--md",
    multiple=True,
    type=str,
    help="Machine description mapping from IDS name to URI, e.g."
    " --md wall=imas:hdf5?path=/path/to/file "
    "--md pf_active=imas:hdf5?path=/path/to/other",
)
@click.option(
    "--automatic-mode",
    default=False,
    is_flag=True,
    help="Automatically discover and visualize time-dependent quantities.",
)
@click.option(
    "--extract-all",
    default=False,
    is_flag=True,
    help="Extract all time-dependent IDS data in automatic mode.",
)
@click.option(
    "--throttle-interval",
    default=0.1,
    show_default=True,
    help="Seconds between UI updates.",
)
def main(
    uri: str,
    md: tuple[str],
    plot_file_path: str,
    port: int,
    automatic_mode: bool,
    extract_all: bool,
    throttle_interval: float,
) -> None:
    """CLI to run the visualization actor as standalone application,
    without needing MUSCLE3.

    Example:

        python cli.py imas:hdf5?path=/path/to/data /path/to/plot_file.py
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    with imas.DBEntry(uri, "r") as entry:
        ids_in_entry = get_available_ids(entry)
        md_dict = create_md_dict(entry, md, ids_in_entry)

        visualization_actor = VisualizationActor(
            plot_file_path=plot_file_path,
            md_dict=md_dict,
            port=port,
            open_browser_on_start=True,
            automatic_mode=automatic_mode,
            extract_all=extract_all,
        )

        feeder_thread = threading.Thread(
            target=feed_data,
            args=(uri, ids_in_entry, visualization_actor, throttle_interval),
            daemon=False,
        )
        logger.info("Waiting for browser to load...")
        time.sleep(3)
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

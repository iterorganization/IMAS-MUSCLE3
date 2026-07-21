"""Recorder actor for MUSCLE3: taps the live traffic of a running workflow
(wire it as an extra receiver on existing conduits) and records each
received IDS to a live-tailable Zarr store, one per occurrence.

See :doc:`/actor_recorder` for usage; shared machinery is in
:mod:`imas_muscle3.recorder`.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from imas import IDSFactory
from libmuscle import Instance, InstanceFlags
from libmuscle.mpp_message import ClosePort
from ymmsl.v0_2 import Operator

from imas_muscle3.recorder.collection import RecorderCollection
from imas_muscle3.recorder.zarr_recorder import ZarrRecorder
from imas_muscle3.utils import (
    get_port_list,
    get_setting_optional,
    ids_name_from_port,
)

logger = logging.getLogger()


@dataclass
class RecorderSettings:
    """Every setting the recorder actor reads, gathered in one place and
    read once per reuse."""

    config: Path
    store_path: Path


def read_settings(instance: Instance) -> RecorderSettings:
    """``store_path`` defaults to the instance's run folder."""
    config = Path(str(instance.get_setting("config", "str")))
    store_path_setting = get_setting_optional(instance, "store_path")
    store_path = (
        Path(str(store_path_setting))
        if store_path_setting is not None
        else Path.cwd()
    )
    return RecorderSettings(config=config, store_path=store_path)


def _precompute_ids_metadata(ids_names: List[str]) -> None:
    """Build each IDS type's metadata once up front."""
    factory = IDSFactory()
    for ids_name in set(ids_names):
        factory.new(ids_name)


def _serve(
    instance: Instance, collection: RecorderCollection, ports: List[str]
) -> Dict[str, BaseException]:
    """Drain every timeline in turn on this thread, round-robin. Senders
    are assumed to keep pace with each other (as in a normal lockstep
    workflow), so blocking on one port in turn doesn't stall the others.
    A timeline ends when its peer's port closes; ``next_timestamp is None``
    is a stream restart (an outer-loop iteration boundary), not the end.

    Returns port -> exception for failed timelines (empty on success);
    the collection is always closed.
    """
    errors: Dict[str, BaseException] = {}
    active = list(ports)
    count = {p: 0 for p in active}
    try:
        while active:
            for port in list(active):
                try:
                    msg, _ = instance._communicator.receive_message(port)
                except (RuntimeError, OSError) as exc:
                    # Peer crashed mid-stream; end this timeline (data so
                    # far is on disk), let the others carry on.
                    active.remove(port)
                    logger.warning(
                        "timeline '%s' ended after %d messages: %r",
                        port,
                        count[port],
                        exc,
                    )
                    continue
                if isinstance(msg.data, ClosePort):
                    active.remove(port)
                    logger.info(
                        "timeline '%s' closed after %d messages",
                        port,
                        count[port],
                    )
                    continue
                try:
                    detail = collection.handle(port, msg)
                except BaseException as exc:  # noqa: B036 -- surfaced to caller
                    errors[port] = exc
                    logger.exception(
                        "timeline '%s' failed after %d messages",
                        port,
                        count[port],
                    )
                    active.remove(port)
                    continue
                count[port] += 1
                logger.info(
                    "handled %s t=%.4e -> %s", port, msg.timestamp, detail
                )
    finally:
        collection.close()
    return errors


def main() -> None:
    """MUSCLE3 execution loop."""
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    while instance.reuse_instance():
        s_ports = get_port_list(instance, Operator.S)
        if not s_ports:
            logger.warning(
                "recorder has no connected S ports; nothing to record."
            )
            break
        # Validate all port -> IDS mappings up front so a bad config fails.
        ids_names = {p: ids_name_from_port(p) for p in s_ports}
        _precompute_ids_metadata(list(ids_names.values()))

        settings = read_settings(instance)
        settings.store_path.mkdir(parents=True, exist_ok=True)
        # Clear stale per-port dirs, never store_path itself (may be the
        # run folder).
        for port in s_ports:
            shutil.rmtree(settings.store_path / port, ignore_errors=True)

        collection = RecorderCollection(
            settings.store_path, settings.config, ids_names, ZarrRecorder
        )
        logger.info(
            "recording %d timeline(s) %s to %s",
            len(s_ports),
            s_ports,
            settings.store_path,
        )
        errors = _serve(instance, collection, s_ports)
        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"recorder timeline(s) failed: {msg}")
            raise RuntimeError(f"recorder timeline(s) failed: {msg}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

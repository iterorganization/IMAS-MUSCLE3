"""Shared machinery for terminal recorder actors.

A recorder is a sink-only MUSCLE3 actor that drains independent *timelines*
(one per connected ``S`` port, dynamic ports), round-robin on a single
thread, and writes each to disk via a pluggable :class:`Sink`. Senders are
assumed to keep pace with each other (as in a normal lockstep workflow), so
blocking on one port in turn doesn't stall the others. A timeline ends when
its peer's port closes; ``next_timestamp is None`` is a stream restart (an
outer-loop iteration boundary), not the end.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Protocol

from imas import IDSFactory
from libmuscle import Instance, InstanceFlags, Message
from libmuscle.mpp_message import ClosePort
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import (
    get_port_list,
    get_setting_optional,
    ids_name_from_port,
)

logger = logging.getLogger()


def precompute_ids_metadata(ids_names: Iterable[str]) -> None:
    """Build each IDS type's metadata once up front."""
    factory = IDSFactory()
    for ids_name in set(ids_names):
        factory.new(ids_name)


class Sink(Protocol):
    """Writes one occurrence's messages to disk; used by one thread only."""

    def write(self, msg: Message) -> str:
        """Write one message; return a short detail to log."""
        ...

    def close(self) -> None:
        """Finalize this occurrence's store."""
        ...


#: Builds a :class:`Sink` from a store base path (no suffix) and an IDS name.
SinkFactory = Callable[[Path, str], Sink]


class OccurrenceRecorder:
    """Records one timeline, one store ``<store_dir>/<NNNN>`` per outer-loop
    iteration, rolling on a stream restart (a message with no
    ``next_timestamp``, or time stepping backwards)."""

    def __init__(
        self, store_dir: Path, ids_name: str, make_sink: SinkFactory
    ) -> None:
        self._store_dir = store_dir
        self._ids_name = ids_name
        self._make_sink = make_sink
        self._occurrence = 0
        self._sink: Optional[Sink] = None
        self._last_time: Optional[float] = None
        self._prev_ended = False

    def handle(self, msg: Message) -> str:
        restarted = self._prev_ended or (
            self._last_time is not None and msg.timestamp < self._last_time
        )
        if self._sink is not None and restarted:
            self._sink.close()
            self._occurrence += 1
            self._sink = None
        if self._sink is None:
            base = self._store_dir / f"{self._occurrence:04d}"
            self._sink = self._make_sink(base, self._ids_name)
        detail = self._sink.write(msg)
        self._last_time = msg.timestamp
        self._prev_ended = msg.next_timestamp is None
        return detail

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()


def serve_timelines(
    recorders: Dict[str, OccurrenceRecorder],
    ids_names: Dict[str, str],
    instance: Instance,
) -> Dict[str, BaseException]:
    """Drain every timeline in turn on this thread, round-robin.

    Returns port -> exception for failed timelines (empty on success);
    recorders are always closed.
    """
    precompute_ids_metadata(ids_names.values())

    errors: Dict[str, BaseException] = {}
    active = list(recorders)
    count = {p: 0 for p in active}
    try:
        while active:
            for port in list(active):
                try:
                    msg, _ = instance._communicator.receive_message(port)
                except (RuntimeError, OSError) as exc:
                    # Peer crashed mid-stream; end this timeline (data so far
                    # is on disk), let the others carry on.
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
                    detail = recorders[port].handle(msg)
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
        for port, recorder in recorders.items():
            try:
                recorder.close()
            except BaseException as exc:  # noqa: B036
                errors.setdefault(port, exc)
                logger.exception(
                    "closing recorder for timeline '%s' failed", port
                )
    return errors


@dataclass
class RecorderSettings:
    """Settings shared by every recorder actor, read once."""

    store_path: Path


def read_recorder_settings(instance: Instance) -> RecorderSettings:
    """``store_path`` defaults to the instance's run folder."""
    store_path_setting = get_setting_optional(instance, "store_path")
    store_path = (
        Path(str(store_path_setting))
        if store_path_setting is not None
        else Path.cwd()
    )
    return RecorderSettings(store_path=store_path)


#: Reads an actor's own settings and returns its per-occurrence SinkFactory.
SinkFactoryBuilder = Callable[[Instance, RecorderSettings], SinkFactory]


def run_recorder(name: str, build_sink_factory: SinkFactoryBuilder) -> None:
    """The shared MUSCLE3 reuse loop for a terminal recorder actor."""
    # Dynamic ports: no port description, ports come from the yMMSL config.
    instance = Instance(flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE)

    while instance.reuse_instance():
        s_ports = get_port_list(instance, Operator.S)
        if not s_ports:
            logger.warning(
                "%s has no connected S ports; nothing to record.", name
            )
            break
        # Validate all port -> IDS mappings up front so a bad config fails.
        ids_names = {p: ids_name_from_port(p) for p in s_ports}

        settings = read_recorder_settings(instance)
        settings.store_path.mkdir(parents=True, exist_ok=True)
        # Clear stale per-port dirs, never store_path itself (may be the
        # run folder).
        for port in s_ports:
            shutil.rmtree(settings.store_path / port, ignore_errors=True)
        sink_factory = build_sink_factory(instance, settings)

        recorders = {
            p: OccurrenceRecorder(
                settings.store_path / p, ids_names[p], sink_factory
            )
            for p in s_ports
        }
        logger.info(
            "%s recording %d timeline(s) %s to %s",
            name,
            len(s_ports),
            s_ports,
            settings.store_path,
        )
        errors = serve_timelines(recorders, ids_names, instance)
        if errors:
            msg = "; ".join(f"{port}: {exc!r}" for port, exc in errors.items())
            instance.error_shutdown(f"{name} timeline(s) failed: {msg}")
            raise RuntimeError(f"{name} timeline(s) failed: {msg}")


def recorder_main(name: str, build_sink_factory: SinkFactoryBuilder) -> None:
    """Recorder module entry point: set up logging, then run the loop."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    run_recorder(name, build_sink_factory)

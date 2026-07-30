"""Recorder actor for MUSCLE3: taps the live traffic of a running workflow
(wire it as an extra receiver on existing conduits) and records each
received IDS to a live-tailable Zarr store, one per occurrence.

See :doc:`/actor_recorder` for usage. All the MUSCLE3 wiring (settings,
checkpoint/resume, multi-port draining) is domain-agnostic and lives in
:func:`muscle3_dashboard.recorder.actor.run_recorder_actor`; this module
supplies only the IMAS-specific piece, deserializing each port's raw bytes
into the IDS its name identifies.
"""

import functools
import logging

from imas_muscle3.recorder.actor import run_recorder_actor
from imas_muscle3.recorder.zarr_recorder import ZarrRecorder
from imas_muscle3.utils import ids_from_message, ids_name_from_port

logger = logging.getLogger()


def main() -> None:
    """MUSCLE3 execution loop."""
    run_recorder_actor(
        deserializer_for_port=lambda port: functools.partial(
            ids_from_message, ids_name_from_port(port)
        ),
        make_recorder=ZarrRecorder,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

"""Distill recorder actor for MUSCLE3: taps the live traffic of a running
workflow (wire it as an extra receiver on existing conduits) and distills
each received IDS to a live-tailable Zarr store, one per occurrence.

See :doc:`/actor_recorder` for usage; shared machinery is in
:mod:`._tap_base`.
"""

from imas_muscle3.actors._tap_base import recorder_main
from imas_muscle3.distill.sink import build_distill_sink_factory

if __name__ == "__main__":
    recorder_main("distill recorder", build_distill_sink_factory)

"""Distill recorder actor for MUSCLE3.

A terminal (sink-only) actor that taps the live traffic of a running workflow
without disturbing the coupling: wire it as an extra receiver on existing
conduits. Every connected ``S`` port is an independent timeline, named after
the IDS it carries. Each received IDS is reduced to plot-ready datasets by the
``config`` file and appended to a live-tailable Zarr store, one store per
outer-loop iteration (occurrence) -- the data behind a muscle3-dashboard
recorder tab.

Example (yMMSL v0.2)::

    components:
      rec:
        implementation: recorder
        ports:
          s: [equilibrium_in, pf_active_in]
    settings:
      rec.config: /path/to/plot_file.py
    implementations:
      recorder:
        executable: python
        args: -u -m imas_muscle3.actors.recorder_component

The drain, occurrence numbering and reuse loop live in :mod:`._tap_base`;
other recording formats (verbatim IMAS, raw messages for debugging) can be
added later as sibling actors on the same base.
"""

from imas_muscle3.actors._tap_base import recorder_main
from imas_muscle3.distill.sink import build_distill_sink_factory

if __name__ == "__main__":
    recorder_main("distill recorder", build_distill_sink_factory)

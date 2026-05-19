import sys


ACTORS = f"""
ymmsl_version: v0.2
description: yMMSL configuration for actors exposed by IMAS-MUSCLE3.
programs:
  source_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.source_component
  sink_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.sink_component
  sink_source_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.sink_source_component
  olc_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.olc_component
  accumulator_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.accumulator_component
  visualization_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.visualization_component
"""
"""yMMSL configuration for all actors exposed by IMAS-MUSCLE3."""

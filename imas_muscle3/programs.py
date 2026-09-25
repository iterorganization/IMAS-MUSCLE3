"""
yMMSL program descriptions of the IMAS-MUSCLE3 actors.

This module is registered as a ``ymmsl.module`` Python entry point (see
``pyproject.toml``), so the actors can be imported in a yMMSL v0.2 file with:

.. code-block:: yaml

    imports:
      - from imas_muscle3.actors import implementation source_component

The programs do not declare ports, so components can use a port for any IDS.
"""

import sys

YMMSL_CONFIG = f"""
ymmsl_version: v0.2
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
  accumulator_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.accumulator_component
  olc_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.olc_component
  visualization_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.visualization_component
"""

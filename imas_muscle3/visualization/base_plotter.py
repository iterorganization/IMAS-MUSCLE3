"""The plotting half of the State/Plotter contract is domain-agnostic (it
only ever touches ``state.data``/``state.variables``), so it lives in
muscle3-dashboard; this re-exports it for existing imas_muscle3 consumers.
"""

from muscle3_dashboard.visualization.base_plotter import BasePlotter

__all__ = ["BasePlotter"]

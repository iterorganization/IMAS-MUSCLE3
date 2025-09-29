"""
Simple example plot which plots the plasma current over time.
"""

import holoviews as hv
import param
import xarray as xr

from imas_muscle3.visualization.base import BasePlotter, BaseState


class State(BaseState):
    pass


class Plotter(BasePlotter):
    pass

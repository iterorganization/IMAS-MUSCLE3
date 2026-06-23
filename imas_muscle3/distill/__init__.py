"""Distillation of IDS streams into compact, append-friendly arrays.

:mod:`~imas_muscle3.distill.distiller` selects each received IDS's
time-dependent 0D/1D/2D quantities and tensorizes them with imas-python's
:func:`imas.util.to_xarray` (so the result follows imas-python's netCDF
conventions); :mod:`~imas_muscle3.distill.zarr_sink` appends those datasets
along ``time`` to a Zarr store. Together they back the
:mod:`~imas_muscle3.actors.distill_component` recorder.
"""

from imas_muscle3.distill.distiller import Distiller
from imas_muscle3.distill.zarr_sink import ZarrSink, group_name

__all__ = ["Distiller", "ZarrSink", "group_name"]

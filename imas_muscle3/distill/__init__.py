"""Distillation of IDS streams into compact, append-friendly arrays.

:mod:`~imas_muscle3.distill.distiller` selects each received IDS's
time-dependent 0D/1D/2D quantities and tensorizes them with imas-python's
:func:`imas.util.to_xarray`; :mod:`~imas_muscle3.distill.zarr_sink` appends
those datasets along ``time`` to a Zarr store;
:mod:`~imas_muscle3.distill.sink` wires them into the ``distill`` format.
"""

from imas_muscle3.distill.distiller import Distiller
from imas_muscle3.distill.zarr_sink import ZarrSink, group_name

__all__ = ["Distiller", "ZarrSink", "group_name"]

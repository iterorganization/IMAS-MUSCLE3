"""Distill IDS streams into compact, live-tailable Zarr stores.

:mod:`~imas_muscle3.distill.sink` turns each received IDS into datasets via
the recorder's config file; :mod:`~imas_muscle3.distill.zarr_sink` appends
them along ``time`` to one Zarr store per occurrence.
"""

from imas_muscle3.distill.zarr_sink import ZarrSink, group_name

__all__ = ["ZarrSink", "group_name"]

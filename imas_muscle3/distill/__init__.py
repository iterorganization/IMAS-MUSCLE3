"""Distill IDS streams into compact, live-tailable Zarr stores."""

from imas_muscle3.distill.zarr_sink import ZarrSink, group_name

__all__ = ["ZarrSink", "group_name"]

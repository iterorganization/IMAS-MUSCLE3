"""The Zarr recorder has no IMAS coupling of its own (deserialization is the
caller's job, via :class:`~imas_muscle3.recorder.base.DeserializeFn`), so it
lives in muscle3-dashboard; this re-exports it for existing imas_muscle3
consumers.
"""

from muscle3_dashboard.recorder.zarr_recorder import (
    ZarrRecorder,
    _combine,
    group_name,
    read_root_attrs,
    write_root_attrs,
)

__all__ = [
    "ZarrRecorder",
    "_combine",
    "group_name",
    "read_root_attrs",
    "write_root_attrs",
]

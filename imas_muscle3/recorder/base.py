"""The recorder base class is domain-agnostic (it deserializes each message
via a caller-supplied function, rather than assuming IMAS), so it lives in
muscle3-dashboard; this re-exports it for existing imas_muscle3 consumers.
"""

from muscle3_dashboard.recorder.base import (
    DeserializeFn,
    ExtractFn,
    Recorder,
    RecorderFactory,
    RecorderState,
)

__all__ = [
    "DeserializeFn",
    "ExtractFn",
    "Recorder",
    "RecorderFactory",
    "RecorderState",
]

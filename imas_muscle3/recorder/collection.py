"""``RecorderCollection`` and its config-loading logic have no IMAS coupling
of their own (a config's ``State``/``extract`` decides that, and messages
arrive pre-deserialized), so they live in muscle3-dashboard; this
re-exports them for existing imas_muscle3 consumers.
"""

from muscle3_dashboard.recorder.collection import (
    LiveState,
    RecorderCollection,
    load_extract_config,
    snapshot_config,
)

__all__ = [
    "LiveState",
    "RecorderCollection",
    "load_extract_config",
    "snapshot_config",
]

"""The recorder actor's MUSCLE3 wiring (settings, checkpoint/resume,
multi-port draining) has no IMAS coupling of its own -- a domain package
supplies only a per-port deserializer -- so it lives in muscle3-dashboard;
this re-exports it for existing imas_muscle3 consumers.
"""

from muscle3_dashboard.recorder.actor import run_recorder_actor

__all__ = ["run_recorder_actor"]

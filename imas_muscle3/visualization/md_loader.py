"""Loads machine-description IDSs for a recorder's dashboard replay view.

A recorder's replay tab has no live MUSCLE3 port to receive machine
description from. ``muscle3_dashboard.components.recorder_viewer`` has no
notion of machine description at all -- it just forwards a recorder's
settings to a plot file's optional ``init_state(settings) -> dict``
function. This module is the IMAS-flavored implementation an IMAS-based
plot file wires up to that hook: it knows the setting is called ``md`` and
how to parse it (``ids_name=imas_uri`` pairs, whitespace-separated).
"""

import logging
from typing import Any, Dict

from imas import DBEntry
from imas.ids_toplevel import IDSToplevel

logger = logging.getLogger()


def load_md(spec: str) -> Dict[str, IDSToplevel]:
    """Parse and load a ``ids_name=imas_uri`` pairs string.

    An entry that fails to parse or load is skipped (logged), not fatal --
    a typo'd machine description shouldn't blank the whole replay tab.
    """
    md: Dict[str, IDSToplevel] = {}
    for entry in (spec or "").split():
        try:
            ids_name, uri = entry.split("=", 1)
        except ValueError:
            logger.warning("could not parse md entry '%s'", entry)
            continue
        try:
            with DBEntry(uri, "r") as db:
                md[ids_name] = db.get(ids_name)
        except Exception:
            logger.warning(
                "could not load md entry '%s'", entry, exc_info=True
            )
    return md


def init_state(settings: Dict[str, Any]) -> Dict[str, IDSToplevel]:
    """``init_state`` hook for the dashboard's recorder viewer: reads this
    recorder's ``md`` setting (if any) and loads it via :func:`load_md`."""
    return load_md(str(settings.get("md", "")))

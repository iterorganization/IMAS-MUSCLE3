"""Combine distilled datasets for one timeline and write them to a Zarr store.

Each distilled IDS (and config group) becomes a Zarr *group* in one store. A
recorder hands the sink one dataset per received message (a single slice, or a
whole trace); the sink buffers them per group and, when the timeline ends,
combines them along ``time`` and writes the group once.

Combining is done with :func:`xarray.concat` (outer join on the ``time``
coordinate), which makes the store robust to the messy realities of real IDS
streams:

* **gaps / inhomogeneous time** — a quantity may be absent on some steps (e.g.
  ``profiles_1d`` empty on TORAX's first solver steps, whose root ``time``
  nonetheless advances). The union of time values is taken and missing entries
  are ``NaN``-filled, so every quantity shares one consistent ``time`` axis.
* **ragged non-time dims** — a profile's length can vary between steps (a
  re-gridded equilibrium); non-time dims are padded with ``NaN`` to the max
  width before concatenation.

Buffering per occurrence trades intra-occurrence live-tailing for a correct,
self-consistent store; occurrences (one per reuse) still appear incrementally.
"""

import logging
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger()

#: The time dimension every distilled dataset shares (see :mod:`.distiller`).
_TIME = "time"


def write_root_attrs(store_path: Path, attrs: Mapping[str, object]) -> None:
    """Stamp metadata onto a store's root group (e.g. the profile reference).

    Lets a reader discover, for instance, which visualization profile produced
    a store. A no-op if the store does not exist (an empty timeline writes no
    store).
    """
    store_path = Path(store_path)
    if not store_path.exists():
        return
    root = zarr.open_group(str(store_path), mode="a")
    root.attrs.update(dict(attrs))


def read_root_attrs(store_path: Path) -> Dict[str, object]:
    """Read a store's root-group metadata (empty dict if unreadable)."""
    try:
        return dict(zarr.open_group(str(store_path), mode="r").attrs)
    except Exception:
        logger.warning("could not read root attrs of %s", store_path, exc_info=True)
        return {}


def group_name(full_path: str) -> str:
    """Zarr-safe group name for a variable's ``ids/path`` key.

    IDS paths use ``/`` as a separator and bare integers for array indices;
    mapping ``/`` to ``.`` yields a flat, readable, collision-free group name
    (DD node names are slash-free identifiers). The original ``full_path`` is
    also kept in the group's ``attrs`` for an exact round-trip.
    """
    return full_path.replace("/", ".")


def _combine(parts: List[xr.Dataset]) -> xr.Dataset:
    """Concatenate one timeline's messages along ``time`` into one dataset.

    Pads ragged non-time dims to their max width, then outer-joins on the
    ``time`` coordinate so gaps become ``NaN`` and every quantity shares one
    ``time``. Non-dimension coordinates are demoted before the concat (xarray
    will not concat a coordinate that is absent from some parts) and restored
    after.
    """
    if len(parts) == 1:
        return parts[0]

    widths: Dict[str, int] = {}
    for part in parts:
        for dim, size in part.sizes.items():
            if dim != _TIME:
                widths[dim] = max(widths.get(dim, 0), size)
    padded = []
    for part in parts:
        pad = {
            dim: (0, widths[dim] - part.sizes[dim])
            for dim in widths
            if dim in part.sizes and part.sizes[dim] < widths[dim]
        }
        padded.append(
            part.pad(pad, constant_values=np.nan) if pad else part
        )

    coord_names = {
        str(c) for part in padded for c in part.coords if str(c) != _TIME
    }
    reset = [part.reset_coords() for part in padded]
    combined = xr.concat(
        reset, dim=_TIME, join="outer", data_vars="all", coords="all"
    )
    return combined.set_coords([c for c in coord_names if c in combined])


class ZarrSink:
    """Buffer one timeline's distilled datasets and write them combined."""

    def __init__(self, store_path: Path) -> None:
        self._store = str(store_path)
        self._buffers: Dict[str, List[xr.Dataset]] = {}

    def append(self, name: str, ds: xr.Dataset) -> None:
        """Buffer a dataset for group ``name``; written combined at :meth:`close`.

        ``ds`` must carry a ``time`` dimension; its length is free — a single
        slice (streamed recording) or a whole trace both work.
        """
        if _TIME not in ds.dims:
            raise ValueError(
                f"{name}: distilled dataset has no '{_TIME}' dimension "
                f"(dims={dict(ds.sizes)})"
            )
        self._buffers.setdefault(group_name(name), []).append(ds)

    def close(self) -> None:
        """Combine each group's buffered messages and write the store."""
        for group, parts in self._buffers.items():
            try:
                _combine(parts).to_zarr(
                    self._store, group=group, mode="w", consolidated=False
                )
            except Exception:
                logger.exception("failed writing group '%s'", group)
        self._buffers.clear()

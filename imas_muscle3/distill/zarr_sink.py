"""Append distilled datasets for one timeline to a Zarr store, live.

Each distilled IDS (and config group) becomes a Zarr *group* in one store. A
recorder hands the sink one dataset per received message (a single slice, or a
whole trace); the sink **writes it to disk immediately**, extending the group
along ``time``, so the store is durable and live-tailable as the run
progresses rather than only once the timeline ends.

A consistent stream (same quantities, same grid every step — the common case)
is a plain ``time`` append. When a message doesn't fit the group's on-disk
schema, the append is rebuilt with :func:`_combine` (:func:`xarray.concat`,
outer join on ``time``) over the existing store plus the new message, which
keeps the store robust to the messy realities of real IDS streams:

* **gaps / inhomogeneous time** — a quantity may be absent on some steps (e.g.
  ``profiles_1d`` empty on TORAX's first solver steps, whose root ``time``
  nonetheless advances). The union of time values is taken and missing entries
  are ``NaN``-filled, so every quantity shares one consistent ``time`` axis.
* **ragged non-time dims** — a profile's length can vary between steps (a
  re-gridded equilibrium); non-time dims are padded with ``NaN`` to the max
  width.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger()

#: The time dimension every distilled dataset shares (see :mod:`.distiller`).
_TIME = "time"


def write_root_attrs(store_path: Path, attrs: Mapping[str, Any]) -> None:
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
        logger.warning(
            "could not read root attrs of %s", store_path, exc_info=True
        )
        return {}


def group_name(full_path: str) -> str:
    """Zarr-safe group name for a variable's ``ids/path`` key.

    IDS paths use ``/`` as a separator and bare integers for array indices;
    mapping ``/`` to ``.`` yields a flat, readable, collision-free group name
    (DD node names are slash-free identifiers). The original ``full_path`` is
    also kept in the group's ``attrs`` for an exact round-trip.
    """
    return full_path.replace("/", ".")


def _signature(ds: xr.Dataset) -> tuple:
    """Schema fingerprint: which quantities, on what non-time grid.

    Two messages with the same signature can be appended along ``time``; a
    different one (a missing quantity, a re-gridded or ragged profile) means
    the group must be rebuilt, since :meth:`xarray.Dataset.to_zarr` does *not*
    reject a mismatched append — it silently corrupts the store.
    """
    names = frozenset(map(str, ds.data_vars))
    dims = tuple(
        sorted((str(d), int(s)) for d, s in ds.sizes.items() if d != _TIME)
    )
    coords = tuple(
        (c, np.asarray(ds[c].values).tobytes())
        for c in sorted(map(str, ds.coords))
        if c != _TIME
    )
    return (names, dims, coords)


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
                widths[str(dim)] = max(widths.get(str(dim), 0), size)
    padded = []
    for part in parts:
        pad = {
            dim: (0, widths[dim] - part.sizes[dim])
            for dim in widths
            if dim in part.sizes and part.sizes[dim] < widths[dim]
        }
        padded.append(part.pad(pad, constant_values=np.nan) if pad else part)

    coord_names = {
        str(c) for part in padded for c in part.coords if str(c) != _TIME
    }
    reset = [part.reset_coords() for part in padded]
    combined = xr.concat(
        reset, dim=_TIME, join="outer", data_vars="all", coords="all"
    )
    return combined.set_coords([c for c in coord_names if c in combined])


class ZarrSink:
    """Append one timeline's distilled datasets to a Zarr store as they arrive.

    Each group's messages are also kept in memory as the source for a rebuild
    (see :meth:`append`); the on-disk store always reflects everything received
    so far, so it is durable and live-tailable mid-run.
    """

    def __init__(self, store_path: Path) -> None:
        self._store = str(store_path)
        self._buffers: Dict[str, List[xr.Dataset]] = {}
        self._sig: Dict[str, tuple] = {}

    def append(self, name: str, ds: xr.Dataset) -> None:
        """Write a dataset for group ``name`` to disk now, along ``time``.

        ``ds`` must carry a ``time`` dimension; its length is free — a single
        slice (streamed recording) or a whole trace both work. The first
        message for a group creates it; a later message with the same schema
        (see :func:`_signature`) is appended along ``time`` (cheap, the common
        streaming case); one with a different schema (a gap, a missing
        quantity, a re-gridded/ragged profile) rebuilds the whole group from
        all of its messages via :func:`_combine`.
        """
        if _TIME not in ds.dims:
            raise ValueError(
                f"{name}: distilled dataset has no '{_TIME}' dimension "
                f"(dims={dict(ds.sizes)})"
            )
        group = group_name(name)
        parts = self._buffers.setdefault(group, [])
        parts.append(ds)
        sig = _signature(ds)
        if len(parts) == 1:
            ds.to_zarr(self._store, group=group, mode="w", consolidated=False)
            self._sig[group] = sig
            return
        if sig == self._sig[group]:
            ds.to_zarr(
                self._store, group=group, append_dim=_TIME, consolidated=False
            )
            return
        # Schema changed: rebuild from all of the group's messages (NaN-filling
        # gaps, padding ragged dims) and rewrite. Clear the group dir first so
        # no stale arrays from the old schema linger.
        shutil.rmtree(Path(self._store) / group, ignore_errors=True)
        try:
            combined = _combine(parts)
            combined.to_zarr(
                self._store, group=group, mode="w", consolidated=False
            )
            self._sig[group] = _signature(combined)
        except Exception:
            logger.exception("failed writing group '%s'", group)

    def close(self) -> None:
        """Every message is already on disk; just drop the rebuild buffer."""
        self._buffers.clear()
        self._sig.clear()

"""Append distilled datasets for one timeline to a Zarr store, live.

Each distilled dataset becomes a Zarr *group*, written to disk immediately and
extended along ``time``, so the store is live-tailable mid-run. A message that
doesn't fit the group's on-disk schema (a gap, a re-gridded/ragged profile)
triggers a rebuild via :func:`_combine`.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger()

#: The time dimension every distilled dataset shares (see :mod:`.sink`).
_TIME = "time"


def write_root_attrs(store_path: Path, attrs: Mapping[str, Any]) -> None:
    """Stamp metadata onto a store's root group. A no-op if the store does
    not exist (an empty timeline writes no store)."""
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
    """Zarr-safe (flat, collision-free) group name for an ``ids/path`` key."""
    return full_path.replace("/", ".")


def _signature(ds: xr.Dataset) -> tuple:
    """Schema fingerprint: which quantities, on what non-time grid. Only a
    matching signature may be appended — ``to_zarr`` does *not* reject a
    mismatched append, it silently corrupts the store."""
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
    """Concat along ``time``, NaN-padding ragged non-time dims and NaN-filling
    gaps. Non-dimension coordinates are demoted first (xarray will not concat
    a coordinate absent from some parts) and restored after."""
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
    """Append one timeline's distilled datasets to a Zarr store as they
    arrive; each group's messages are kept in memory as rebuild source."""

    def __init__(self, store_path: Path) -> None:
        self._store = str(store_path)
        self._buffers: Dict[str, List[xr.Dataset]] = {}
        self._sig: Dict[str, tuple] = {}

    def append(self, name: str, ds: xr.Dataset) -> None:
        """Write a dataset (single slice or whole trace, must carry ``time``)
        for group ``name`` to disk now: appended along ``time`` if its schema
        matches the group's, else the group is rebuilt via :func:`_combine`.
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
        # Schema changed: clear the group dir (no stale arrays from the old
        # schema) and rewrite it from all messages.
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

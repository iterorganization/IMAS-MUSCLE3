"""The Zarr :class:`~imas_muscle3.recorder.base.Recorder`: appends one
port's extracted datasets to a Zarr store, live-tailable mid-run.

Each occurrence gets its own store, ``<store_dir>/<occurrence>.zarr``; every
extracted dataset becomes a Zarr *group* within it, written to disk
immediately and extended along ``time``. A message that doesn't fit the
group's on-disk schema (a gap, a re-gridded/ragged profile) triggers a
rebuild via :func:`_combine`.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import xarray as xr
import zarr

from imas_muscle3.recorder.base import Recorder

logger = logging.getLogger()

#: The time dimension every extracted dataset shares (see :mod:`.collection`).
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
    matching signature may be appended -- ``to_zarr`` does *not* reject a
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


class ZarrRecorder(Recorder):
    """Appends one port's extracted datasets to a Zarr store per occurrence;
    each message's schema is kept in memory as rebuild source until close."""

    _store: str
    _buffers: Dict[str, List[xr.Dataset]]
    _sig: Dict[str, tuple]

    def _open_occurrence(self, base: Path) -> None:
        self._store = str(base.with_suffix(".zarr"))
        self._buffers = {}
        self._sig = {}

    def _write(self, datasets: Dict[str, xr.Dataset]) -> str:
        for name, ds in datasets.items():
            self._append(name, ds)
        return f"{len(datasets)} dataset(s)"

    def _append(self, name: str, ds: xr.Dataset) -> None:
        if _TIME not in ds.dims:
            raise ValueError(
                f"{name}: extracted dataset has no '{_TIME}' dimension "
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

    def _close_occurrence(self) -> None:
        self._buffers.clear()
        self._sig.clear()
        # Lets a viewer group stores and load the matching config.
        write_root_attrs(
            Path(self._store),
            {
                "occurrence": int(Path(self._store).stem),
                "distill_profile": self._profile,
            },
        )

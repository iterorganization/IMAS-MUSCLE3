"""Append distilled single-time datasets to a Zarr store, one timeline per store.

Each distilled variable becomes a Zarr *group* inside one store; new time slices
are appended along the ``time`` dimension. The store is the natural fit for this
shape: appends are cheap, non-time dims are chunked, and a viewer can read the
growing arrays while the run is still writing them.

**Ragged non-time dims.** Zarr arrays have a fixed shape on their non-append
dims, but a profile's length can vary between time slices (e.g. a re-gridded
equilibrium). The first slice of a variable fixes that width; later slices are
padded with ``NaN`` (shorter) or truncated (longer, with a warning) to match.
A fully ragged representation is future work; pad/truncate keeps the common
fixed-grid case exact and never crashes on the rest.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr

logger = logging.getLogger()

#: The append dimension; every distilled dataset carries it (see :mod:`.distiller`).
_TIME = "time"


def group_name(full_path: str) -> str:
    """Zarr-safe group name for a variable's ``ids/path`` key.

    IDS paths use ``/`` as a separator and bare integers for array indices;
    mapping ``/`` to ``.`` yields a flat, readable, collision-free group name
    (DD node names are slash-free identifiers). The original ``full_path`` is
    also kept in the group's ``attrs`` for an exact round-trip.
    """
    return full_path.replace("/", ".")


class ZarrSink:
    """Append-along-time writer for one timeline's distilled variables."""

    def __init__(self, store_path: Path) -> None:
        self._store = str(store_path)
        # Established non-time dim sizes per group, from its first slice.
        self._widths: Dict[str, Dict[str, int]] = {}

    def append(self, name: str, ds: xr.Dataset) -> None:
        """Append a single-time dataset to its group (``name``)."""
        if ds.sizes.get(_TIME) != 1:
            raise ValueError(
                f"{name}: expected a single-time dataset, got "
                f"{_TIME}={ds.sizes.get(_TIME)}"
            )
        group = group_name(name)
        if group not in self._widths:
            self._widths[group] = {
                dim: size for dim, size in ds.sizes.items() if dim != _TIME
            }
            ds.to_zarr(self._store, group=group, mode="w", consolidated=False)
            return
        ds = self._reconcile(group, name, ds)
        ds.to_zarr(
            self._store, group=group, append_dim=_TIME, consolidated=False
        )

    def _reconcile(self, group: str, name: str, ds: xr.Dataset) -> xr.Dataset:
        """Pad/truncate ``ds`` non-time dims to the group's established sizes."""
        for dim, want in self._widths[group].items():
            have = ds.sizes.get(dim)
            if have is None or have == want:
                continue
            if have < want:
                ds = ds.pad({dim: (0, want - have)}, constant_values=np.nan)
            else:
                logger.warning(
                    "%s: slice %s=%d exceeds stored width %d; truncating",
                    name,
                    dim,
                    have,
                    want,
                )
                ds = ds.isel({dim: slice(0, want)})
        return ds

    def close(self) -> None:  # no open handles to release; appends are atomic
        pass

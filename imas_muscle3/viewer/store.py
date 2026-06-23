"""Discover and read the distilled Zarr stores written by the distill recorder.

A run produces one store per timeline (``<port>.zarr``), each with a group per
IDS (and any config groups). The arrays follow imas-python's ``to_xarray``
conventions, so a variable's dims are ``(time,)`` for a scalar, ``(time, X)``
for a profile and ``(time, X, Y)`` for a map, with ``X``/``Y`` coordinates named
after their DD path and listed in the CF ``coordinates`` attribute.

These helpers are deliberately free of Panel/holoviews so they can be unit
tested directly; plotting lives in :mod:`imas_muscle3.viewer.plots`.
"""

import logging
from pathlib import Path
from typing import List

import xarray as xr
import zarr

logger = logging.getLogger(__name__)

#: The append/time dimension shared by every distilled variable.
TIME = "time"


def find_stores(run_dir: Path) -> List[Path]:
    """All distilled Zarr stores under ``run_dir`` (``*.zarr`` with a root node).

    Stores can sit anywhere below the run directory (by default each actor
    writes into its own ``instances/<name>/workdir``), so the search recurses.
    """
    run_dir = Path(run_dir)
    stores = [
        p
        for p in run_dir.rglob("*.zarr")
        if p.is_dir() and (p / "zarr.json").exists()
    ]
    return sorted(set(stores))


def list_groups(store: Path) -> List[str]:
    """Names of the groups (one per recorded IDS) in a store."""
    try:
        root = zarr.open_group(str(store), mode="r")
    except Exception:
        logger.warning("could not open zarr store %s", store, exc_info=True)
        return []
    return sorted(name for name, _ in root.groups())


def open_group(store: Path, group: str) -> xr.Dataset:
    """Open one group as an xarray Dataset (re-open to see appended times)."""
    return xr.open_zarr(store, group=group, consolidated=False)


def plottable_variables(ds: xr.Dataset) -> List[str]:
    """Data variables that vary over time (i.e. that we can plot)."""
    return sorted(str(v) for v in ds.data_vars if TIME in ds[v].dims)


def non_time_dims(da: xr.DataArray) -> List[str]:
    """The variable's dimensions other than time, in order."""
    return [str(d) for d in da.dims if d != TIME]


def variable_rank(da: xr.DataArray) -> int:
    """0 for a scalar-over-time, 1 for a profile, 2 for a map."""
    return len(non_time_dims(da))


def coord_names(da: xr.DataArray) -> List[str]:
    """Name of a coordinate spanning each of ``da``'s non-time dims, in order.

    The CF ``coordinates`` attribute that :func:`imas.util.to_xarray` writes is
    consumed by xarray on save (it tracks the association structurally), so we
    recover it from the DataArray's own coordinates: a non-time coordinate whose
    only non-time dimension is ``dim`` is that dimension's axis. Dimensions with
    no such coordinate are omitted, so a result shorter than ``non_time_dims``
    means the plotter should fall back to integer index positions.
    """
    names: List[str] = []
    for dim in non_time_dims(da):
        for cname, cvar in da.coords.items():
            if str(cname) == TIME:
                continue
            if [str(d) for d in cvar.dims if d != TIME] == [dim]:
                names.append(str(cname))
                break
    return names

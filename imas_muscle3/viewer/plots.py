"""Holoviews plots for distilled variables, dispatched by rank.

A distilled variable is plotted according to its shape (see
:mod:`imas_muscle3.viewer.store`):

* **scalar** ``(time,)`` → a curve of the value over time;
* **profile** ``(time, X)`` → the profile at the selected time, against its
  coordinate;
* **map** ``(time, X, Y)`` → a filled quadmesh at the selected time.

Axis labels use the DD ``units`` and coordinate names carried in the store.
"""

import logging
from typing import Optional

import holoviews as hv
import numpy as np
import xarray as xr

from imas_muscle3.viewer.store import TIME, variable_rank

logger = logging.getLogger(__name__)


def _label(name: str, units: Optional[str]) -> str:
    return f"{name} [{units}]" if units else name


def _axis(
    ds: xr.Dataset, var: str, dim: str, time_index: int
) -> tuple[str, np.ndarray]:
    """The label and values of ``dim``'s axis at a time index.

    Distilled coordinates are stored per time step (dims ``(time, <dim>)``), so
    we select the current time; a dimension with no named coordinate falls back
    to integer positions labelled by the dimension name.
    """
    da = ds[var]
    for cname, cvar in da.coords.items():
        if str(cname) == TIME:
            continue
        if [str(d) for d in cvar.dims if d != TIME] == [dim]:
            return str(cname), np.asarray(
                ds[cname].isel({TIME: time_index}).values
            )
    return dim, np.arange(ds.sizes[dim])


def plot_variable(ds: xr.Dataset, var: str, time_index: int) -> hv.Element:
    """Plot ``var`` from ``ds`` at ``time_index`` (clamped to the data)."""
    da = ds[var]
    n_time = ds.sizes[TIME]
    time_index = max(0, min(time_index, n_time - 1))
    units = da.attrs.get("units")
    rank = variable_rank(da)

    if rank == 0:
        times = np.asarray(ds[TIME].values)
        return hv.Curve(
            (times, np.asarray(da.values)), kdims=["time [s]"], vdims=[var]
        ).opts(title=var, ylabel=_label(var, units), responsive=True)

    t = float(ds[TIME].values[time_index])
    if rank == 1:
        (dim,) = [d for d in da.dims if d != TIME]
        xlabel, x = _axis(ds, var, dim, time_index)
        y = np.asarray(da.isel({TIME: time_index}).values)
        return hv.Curve((x, y), kdims=[xlabel], vdims=[var]).opts(
            title=f"{var} (t={t:.3f}s)",
            ylabel=_label(var, units),
            responsive=True,
        )

    # rank 2: a map at the selected time
    dim0, dim1 = [d for d in da.dims if d != TIME]
    ylabel, y = _axis(ds, var, dim0, time_index)
    xlabel, x = _axis(ds, var, dim1, time_index)
    z = np.asarray(da.isel({TIME: time_index}).values)
    return hv.QuadMesh((x, y, z), kdims=[xlabel, ylabel], vdims=[var]).opts(
        title=f"{var} (t={t:.3f}s)",
        colorbar=True,
        cmap="viridis",
        responsive=True,
    )

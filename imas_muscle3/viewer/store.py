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
from typing import List, Optional

import xarray as xr
import zarr

from imas_muscle3.distill.zarr_sink import read_root_attrs

logger = logging.getLogger(__name__)

#: The append/time dimension shared by every distilled variable.
TIME = "time"


def store_profile(store: Path) -> Optional[str]:
    """Path of the visualization profile that produced this store, if any.

    The distill recorder stamps it into the store's root attrs when run with a
    ``config``; ``None`` means a plain auto-distilled store (generic view).
    """
    profile = read_root_attrs(store).get("distill_profile")
    return str(profile) if profile else None


def store_port(store: Path) -> str:
    """The port (timeline) a store belongs to — its parent directory name.

    Stores are laid out ``<base>/<port>/<NNNN>.zarr`` (one occurrence per reuse).
    """
    return store.parent.name


def store_occurrence(store: Path) -> str:
    """The occurrence label of a store (the reuse / F_INIT-loop index).

    Read from the root attrs the recorder stamps, falling back to the file stem.
    """
    occ = read_root_attrs(store).get("occurrence")
    return f"{int(occ):04d}" if occ is not None else store.stem


def store_instance(store: Path) -> Optional[str]:
    """The MUSCLE instance that wrote a store, from a run layout if present.

    Stores under ``.../instances/<name>/workdir/...`` (where a recorder's
    default ``store_path`` is its own working directory) yield ``<name>``; this
    keeps stores from two recorders that share a port name distinct. ``None``
    when the path doesn't follow that layout (e.g. an explicit ``store_path``).
    """
    parts = store.parts
    if "instances" in parts:
        i = parts.index("instances")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def store_label(store: Path) -> str:
    """A unique, readable label: ``[<instance>/]<port>/<occurrence>``."""
    base = f"{store_port(store)}/{store_occurrence(store)}"
    instance = store_instance(store)
    return f"{instance}/{base}" if instance else base


def occurrences(run_dir: Path) -> dict:
    """Map each occurrence label to its ``{port: store path}`` across the run.

    Lets a profile view scope to one F_INIT loop: all ports recorded at the same
    occurrence, side by side.
    """
    grouped: dict = {}
    for store in find_stores(run_dir):
        grouped.setdefault(store_occurrence(store), {})[store_port(store)] = store
    return dict(sorted(grouped.items()))


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

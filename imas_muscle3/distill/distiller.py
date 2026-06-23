"""Select distillable quantities from an IDS and tensorize them with imas-python.

The :class:`Distiller` is the panel-free distillation core behind the
:mod:`~imas_muscle3.actors.distill_component` recorder. For each received
single-time-slice IDS it returns a mapping ``group -> xarray.Dataset`` ready to
append along ``time``.

The datasets are built by **imas-python's own** :func:`imas.util.to_xarray`, so
they follow exactly the conventions of imas-python's netCDF backend and
``DBEntry.get(...).to_xarray()``:

* variable names are the DD path with ``/`` replaced by ``.``
  (``profiles_1d.electrons.density``);
* ``time`` is a real coordinate dimension (length 1 per received slice), so
  slices concatenate cleanly;
* each quantity carries its DD ``units``, ``documentation`` and a CF-style
  ``coordinates`` attribute, and its coordinate arrays are proper dataset
  coordinates (``profiles_1d.grid.rho_tor_norm``);
* non-time dimensions are named after their coordinate path (``…:i``) and float
  gaps are filled with ``NaN``.

The distiller's only job is to *choose which paths* to tensorize:

* **auto-discovery** (default): every time-dependent 0D/1D/2D ``FLT`` quantity,
  found once per IDS type by walking the tree (GGD/grid subtrees are skipped, as
  they explode into tens of thousands of nodes). All of them are tensorized into
  one dataset keyed by the IDS name.
* an optional **config callable** ``extract(ids) -> dict[str, xarray.Dataset]``
  for derived/geometric quantities (separatrix, contours, ...) that
  auto-discovery cannot express. Its datasets are recorded as extra groups and
  must likewise hold a single ``time`` step.
"""

import logging
from typing import Callable, Dict, Iterator, List, Optional, Set

import imas
import numpy as np
import xarray as xr
from imas.ids_base import IDSBase
from imas.ids_data_type import IDSDataType
from imas.ids_metadata import IDSType
from imas.ids_primitive import IDSPrimitive
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

logger = logging.getLogger()

#: A config callable mapping an IDS to extra single-time datasets.
ExtractFn = Callable[[IDSToplevel], Dict[str, xr.Dataset]]

#: Highest array rank (excluding time) we auto-distill; GGDs aside, 0/1/2D
#: covers scalars, profiles and maps — what a dashboard can plot.
_MAX_NDIM = 2

#: The uniform append/time dimension every distilled dataset is normalized to.
_TIME = "time"


def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
    """Rename a dataset's primary time-like dimension to a uniform ``time``.

    :func:`imas.util.to_xarray` names the time axis ``time`` for a
    homogeneous-time IDS but ``<aos>.time`` (e.g. ``time_slice.time``) for a
    heterogeneous one. Either way — and whether the IDS holds a single slice or
    a whole trace — we want one ``time`` dimension so a sink can append along it
    and a viewer can find it. A dataset with no time-like axis is returned
    unchanged.

    A heterogeneous IDS can carry *several* time axes (e.g. equilibrium has both
    ``time_slice.time`` and ``grids_ggd.time``); only one can become ``time``, so
    we pick the axis the most data variables actually use (the dominant one),
    leaving the minor axes as-is. If ``time`` already exists, keep it.
    """
    if _TIME in ds.dims:
        return ds
    timelike = [str(d) for d in ds.dims if str(d).endswith(".time")]
    if not timelike:
        return ds
    if len(timelike) > 1:
        timelike = [_dominant_time_axis(ds, timelike)]
    return ds.rename({timelike[0]: _TIME})


def _dominant_time_axis(ds: xr.Dataset, timelike: List[str]) -> str:
    """The time axis the most data variables use (ties broken by length)."""
    usage = {
        d: sum(1 for v in ds.data_vars if d in ds[v].dims) for d in timelike
    }
    return max(timelike, key=lambda d: (usage[d], ds.sizes[d]))


def _divergent_time_axes(ds: xr.Dataset) -> List[str]:
    """The ``*.time`` axes whose values differ from the dominant one.

    Returns ``[]`` when there is a single time axis, or when several coincide
    (e.g. equilibrium's ``time_slice.time`` and ``grids_ggd.time`` hold the same
    grid) — those are harmless. A non-empty result means genuinely inhomogeneous
    time: quantities on different axes won't share one ``time``.
    """
    timelike = [str(d) for d in ds.dims if str(d).endswith(".time")]
    if len(timelike) < 2:
        return []
    ref = _dominant_time_axis(ds, timelike)
    ref_vals = np.asarray(ds[ref].values) if ref in ds.coords else None
    divergent = []
    for d in timelike:
        if d == ref or d not in ds.coords:
            continue
        vals = np.asarray(ds[d].values)
        same = ref_vals is not None and vals.shape == ref_vals.shape and (
            np.allclose(vals, ref_vals)
        )
        if not same:
            divergent.append(d)
    return divergent

#: structure_reference values whose subtrees are skipped (too large to distill).
_SKIP_STRUCTURES = frozenset(
    {"generic_grid_dynamic", "generic_grid_aos3_root", "grid"}
)


class Distiller:
    """Pick distillable quantities and tensorize them with imas-python."""

    def __init__(
        self, auto: bool = True, extract: Optional[ExtractFn] = None
    ) -> None:
        if not auto and extract is None:
            raise ValueError(
                "Distiller with auto=False needs an extract config; "
                "otherwise it would record nothing."
            )
        self._auto = auto
        self._extract = extract
        # DD paths to tensorize per IDS name, discovered lazily on first sight.
        self._paths: Dict[str, List[str]] = {}
        # IDS names whose time axes we have already vetted (warn at most once).
        self._time_checked: Set[str] = set()

    def distill(self, ids: IDSToplevel) -> Dict[str, xr.Dataset]:
        """Return ``group -> Dataset`` for one received IDS.

        ``ids`` may hold a single time slice (as a source streams them) or a
        whole trace (as the inverse pipeline passes per Picard iteration); both
        are tensorized as-is and their time-like axis normalized to ``time``
        (see :func:`_normalize_time`), so the result has a length-1 or
        length-N ``time`` dimension respectively. A sink appends single slices
        along ``time``; a whole-trace dataset is one self-contained occurrence.

        The auto-discovered quantities are tensorized into one dataset keyed by
        the IDS name; the optional config's datasets are added as extra groups.
        On a key clash the config wins (it is the explicit intent).
        """
        out: Dict[str, xr.Dataset] = {}
        if self._auto:
            ids_name = ids.metadata.name
            if ids_name not in self._paths:
                self._paths[ids_name] = self._discover(ids)
            paths = self._paths[ids_name]
            if paths:
                ds = imas.util.to_xarray(ids, *paths)
                self._warn_if_time_axes_diverge(ids_name, ds)
                out[ids_name] = ds
        if self._extract is not None:
            out.update(self._extract(ids))
        return {name: _normalize_time(ds) for name, ds in out.items()}

    def paths(self, ids_name: str) -> List[str]:
        """DD paths discovered so far for an IDS name (after ``distill``)."""
        return list(self._paths.get(ids_name, []))

    def _warn_if_time_axes_diverge(
        self, ids_name: str, ds: xr.Dataset
    ) -> None:
        """Warn once per IDS on *genuinely* inhomogeneous time.

        Heterogeneous IDSs can carry several ``*.time`` axes; usually they hold
        the same grid (e.g. equilibrium's ``time_slice.time`` and
        ``grids_ggd.time``) and are harmless — the dominant one becomes ``time``.
        We warn only when the axes actually differ, since then quantities on the
        minor axes are recorded against a ``time`` that isn't theirs.
        """
        if ids_name in self._time_checked:
            return
        self._time_checked.add(ids_name)
        divergent = _divergent_time_axes(ds)
        if divergent:
            logger.warning(
                "IDS '%s' has inhomogeneous time: axes %s differ from the "
                "dominant time axis; their quantities are recorded against a "
                "'time' that is not theirs. Prefer homogeneous_time output.",
                ids_name,
                divergent,
            )

    # --- auto-discovery -----------------------------------------------------

    def _tree_iter(self, node: IDSBase) -> Iterator[IDSBase]:
        """Yield leaf primitives, skipping GGD/grid subtrees (too large)."""
        if isinstance(node, IDSPrimitive):
            return
        iterator = (
            node.iter_nonempty_() if isinstance(node, IDSStructure) else node
        )
        for child in iterator:
            structure_reference = getattr(
                child.metadata, "structure_reference", None
            )
            if (
                structure_reference in _SKIP_STRUCTURES
                or child.metadata.name == "ggd"
            ):
                continue
            if isinstance(child, IDSPrimitive):
                yield child
            else:
                yield from self._tree_iter(child)

    def _discover(self, ids: IDSToplevel) -> List[str]:
        """DD path strings of time-dependent 0D/1D/2D FLT quantities in ``ids``.

        Paths are the index-free ``metadata.path_string`` (so the same quantity
        across array-of-structure elements collapses to one path), which is what
        :func:`imas.util.to_xarray` expects; tensorization turns the
        array-of-structures over time into the ``time`` dimension.
        """
        paths: Set[str] = set()
        for node in self._tree_iter(ids):
            metadata = node.metadata
            if (
                metadata.data_type != IDSDataType.FLT
                or metadata.ndim > _MAX_NDIM
                or metadata.type != IDSType.DYNAMIC
            ):
                continue
            path = metadata.path_string
            if path == "time":
                continue
            paths.add(path)

        ordered = sorted(paths)
        logger.info(
            "discovered %d distillable path(s) in IDS '%s'",
            len(ordered),
            ids.metadata.name,
        )
        return ordered

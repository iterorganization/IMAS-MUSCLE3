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
import xarray as xr
from imas.ids_base import IDSBase
from imas.ids_data_type import IDSDataType
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
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

    def distill(
        self, ids: IDSToplevel, time: Optional[float] = None
    ) -> Dict[str, xr.Dataset]:
        """Return ``group -> single-time Dataset`` for one received IDS.

        ``ids`` must hold a single time slice (as the source streams them).
        It is canonicalized to one *homogeneous* time point (``time`` if given,
        else ``ids.time[0]``) so :func:`imas.util.to_xarray` yields a ``time``
        dimension every quantity shares; a heterogeneous IDS would otherwise get
        a separate per-array-of-structure time axis (``time_slice.time``) that
        cannot be appended on uniformly.

        The auto-discovered quantities are tensorized into one dataset keyed by
        the IDS name; the optional config's datasets are added as extra groups.
        On a key clash the config wins (it is the explicit intent).
        """
        self._canonicalize_time(ids, time)
        out: Dict[str, xr.Dataset] = {}
        if self._auto:
            ids_name = ids.metadata.name
            if ids_name not in self._paths:
                self._paths[ids_name] = self._discover(ids)
            paths = self._paths[ids_name]
            if paths:
                out[ids_name] = imas.util.to_xarray(ids, *paths)
        if self._extract is not None:
            out.update(self._extract(ids))
        return out

    @staticmethod
    def _canonicalize_time(ids: IDSToplevel, time: Optional[float]) -> None:
        """Coerce a single-slice IDS to one homogeneous time point."""
        if time is None:
            time = float(ids.time[0]) if len(ids.time) else 0.0
        ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
        ids.time = [time]

    def paths(self, ids_name: str) -> List[str]:
        """DD paths discovered so far for an IDS name (after ``distill``)."""
        return list(self._paths.get(ids_name, []))

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

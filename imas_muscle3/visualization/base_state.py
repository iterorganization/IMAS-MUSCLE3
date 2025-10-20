import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import imas
import numpy as np
import panel as pn
import param
import xarray as xr
from imas.ids_data_type import IDSDataType
from imas.ids_metadata import IDSType
from imas.ids_primitive import IDSNumericArray
from imas.ids_toplevel import IDSToplevel
from imas.util import tree_iter

logger = logging.getLogger()


class Dim(Enum):
    """Enum for variable dimensionality."""

    ZERO_D = "0D"
    ONE_D = "1D"
    TWO_D = "2D"


@dataclass
class Variable:
    """Represents a single discoverable variable from an IDS."""

    ids_name: str
    path: str
    dimension: Dim
    coord_names: List[str] = field(default_factory=list)
    is_visualized: bool = False

    @property
    def full_path(self) -> str:
        """Returns the full path for UI display (ids_name/path)."""
        return f"{self.ids_name}/{self.path}"


class BaseState(param.Parameterized):
    """Abstract container for simulation state. Holds live simulation data
    as well as data from a machine description.
    """

    data = param.Dict(
        default={}, doc="Mapping of IDS name to live IDS data objects."
    )
    md = param.Dict(
        default={},
        doc="Mapping of IDS name to machine description data objects.",
    )
    variables = param.Dict(
        default={},
        doc=("Mapping of a variable's full path to a Variable object"),
    )

    def __init__(
        self,
        md_dict: Dict[str, IDSToplevel],
        auto=False,
        extract_all: bool = False,
    ) -> None:
        super().__init__()
        self.extract_all = extract_all
        self.auto = auto
        self.md = md_dict
        self._discovery_done = set()

    def _get_coord_name(self, path: str, i: int, coord_obj) -> str:
        """Helper to get a coordinate name from metadata or generate one."""
        if isinstance(coord_obj, IDSNumericArray):
            return coord_obj.metadata.name
        return f"{path}_coord{i}"

    def _discover_variables(self, ids: IDSToplevel):
        """Discovers numerical variables in an IDS and populates the state.

        Args:
            ids: The IDS to discover variables for.
        """
        ids_name = ids.metadata.name
        logger.info(f"Discovering float variables in IDS '{ids_name}'...")
        new_variables = {}
        for node in tree_iter(ids, leaf_only=True):
            metadata = node.metadata
            # Only discover time-dependent 0D, 1D and 2D FLT quantities
            if (
                metadata.data_type != IDSDataType.FLT
                or metadata.ndim > 2
                or metadata.type != IDSType.DYNAMIC
            ):
                continue
            path = str(imas.util.get_full_path(node))
            if path == "time":
                continue

            full_path = f"{ids_name}/{path}"
            dim = Dim.ZERO_D
            coord_names = []

            if metadata.ndim == 1:
                # Check if it's a 0D variable over time
                if not (
                    hasattr(node.coordinates[0], "metadata")
                    and node.coordinates[0].metadata.name == "time"
                ):
                    dim = Dim.ONE_D
                    coord_names = [
                        self._get_coord_name(path, 0, node.coordinates[0])
                    ]
            elif metadata.ndim == 2:
                dim = Dim.TWO_D
                coord_names = [
                    self._get_coord_name(path, 0, node.coordinates[0]),
                    self._get_coord_name(path, 1, node.coordinates[1]),
                ]

            new_variables[full_path] = Variable(
                ids_name=ids_name,
                path=path,
                dimension=dim,
                coord_names=coord_names,
            )

            # FIXME: panel crashes when too many quantities are discovered
            if len(new_variables) >= 1000:
                msg = (
                    "Cannot automatically discover more than 1000 quantities, "
                    "only the first 1000 are shown"
                )
                pn.state.notifications.error(msg)
                logger.error(msg)
                break

        self.variables.update(new_variables)
        self.param.trigger("variables")
        self._discovery_done.add(ids_name)
        logger.info(
            f"Discovered {len(new_variables)} variables in IDS '{ids_name}'."
        )

    def extract_data(self, ids):
        if self.auto:
            self.automatic_extract(ids)
        self.extract(ids)

    def extract(self, ids: IDSToplevel) -> None:
        """Extract data from an IDS and store it into a state object. Must be
        implemented by subclasses.

        Args:
            ids: An IDS containing simulation results.
        """
        raise NotImplementedError(
            "A state class needs to implement an `extract` method"
        )

    def automatic_extract(self, ids: IDSToplevel) -> None:
        """Extracts data for visualized variables from the given IDS.

        Args:
            ids: The IDS to extract data from.
        """
        ids_name = ids.metadata.name
        if ids_name not in self._discovery_done:
            self._discover_variables(ids)

        if self.extract_all:
            vars_to_extract = [
                var
                for var in self.variables.values()
                if var.ids_name == ids_name
            ]
        else:
            vars_to_extract = [
                var
                for var in self.variables.values()
                if var.ids_name == ids_name and var.is_visualized
            ]

        for var in vars_to_extract:
            if var.dimension == Dim.ZERO_D:
                self._extract_0d(ids, var)
            elif var.dimension == Dim.ONE_D:
                self._extract_1d(ids, var)
            elif var.dimension == Dim.TWO_D:
                self._extract_2d(ids, var)

    def _extract_0d(self, ids: IDSToplevel, var: Variable):
        """Extracts and stores 0D data.

        Args:
            ids: The ids to extract data from.
            var: The variable to extract.
        """
        current_time = float(ids.time[0])
        value_obj = ids[var.path]
        value = (
            float(value_obj.value)
            if value_obj.metadata.ndim == 0
            else float(value_obj[0])
        )

        new_ds = xr.Dataset(
            {var.full_path: ("time", [value])}, coords={"time": [current_time]}
        )
        if var.full_path in self.data:
            self.data[var.full_path] = xr.concat(
                [self.data[var.full_path], new_ds], dim="time"
            )
        else:
            self.data[var.full_path] = new_ds

    def _extract_1d(self, ids: IDSToplevel, var: Variable):
        """Extracts and stores 1D data.

        Args:
            ids: The ids to extract data from.
            var: The variable to extract.
        """
        current_time = float(ids.time[0])
        value_obj = ids[var.path]
        arr = np.array(value_obj[:], dtype=float)
        coords_obj = value_obj.coordinates[0]
        coord_name = var.coord_names[0]
        coords = np.array(coords_obj, dtype=float)

        new_ds = xr.Dataset(
            {
                var.full_path: (("time", "i"), arr[np.newaxis, :]),
                f"{var.full_path}_{coord_name}": (
                    ("time", "i"),
                    coords[np.newaxis, :],
                ),
            },
            coords={"time": [current_time]},
        )
        if var.full_path in self.data:
            self.data[var.full_path] = xr.concat(
                [self.data[var.full_path], new_ds], dim="time"
            )
        else:
            self.data[var.full_path] = new_ds

    def _extract_2d(self, ids: IDSToplevel, var: Variable):
        """Extracts and stores 2D data.

        Args:
            ids: The ids to extract data from.
            var: The variable to extract.
        """
        current_time = float(ids.time[0])
        value_obj = ids[var.path]
        arr = np.array(value_obj[:], dtype=float)
        coords_obj0 = value_obj.coordinates[0]
        coords_obj1 = value_obj.coordinates[1]
        coords0 = np.array(coords_obj0, dtype=float)
        coords1 = np.array(coords_obj1, dtype=float)

        new_ds = xr.Dataset(
            {
                var.full_path: (("time", "y", "x"), arr[np.newaxis, :, :]),
                f"{var.full_path}_{var.coord_names[0]}": (
                    ("time", "y"),
                    coords0[np.newaxis, :],
                ),
                f"{var.full_path}_{var.coord_names[1]}": (
                    ("time", "x"),
                    coords1[np.newaxis, :],
                ),
            },
            coords={"time": [current_time]},
        )
        if var.full_path in self.data:
            self.data[var.full_path] = xr.concat(
                [self.data[var.full_path], new_ds], dim="time"
            )
        else:
            self.data[var.full_path] = new_ds

import functools
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import holoviews as hv
import imas
import numpy as np
import panel as pn
import param
import xarray as xr
from imas.ids_data_type import IDSDataType
from imas.ids_primitive import IDSNumericArray
from imas.ids_toplevel import IDSToplevel
from imas.util import tree_iter
from panel.viewable import Viewable

from imas_muscle3.visualization.base import BasePlotter, BaseState
from imas_muscle3.visualization.resizable_float_panel import (
    ResizableFloatPanel,
)

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


class State(BaseState):
    variables = param.Dict(
        default={},
        doc=("Mapping of a variable's full path to a Variable object"),
    )

    def __init__(self, md_dict: Dict[str, IDSToplevel], **kwargs) -> None:
        super().__init__(md_dict, **kwargs)
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
            if metadata.data_type != IDSDataType.FLT or metadata.ndim > 2:
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

        self.variables.update(new_variables)
        self.param.trigger("variables")
        self._discovery_done.add(ids_name)
        logger.info(
            f"Discovered {len(new_variables)} variables in IDS '{ids_name}'."
        )

    def extract(self, ids: IDSToplevel) -> None:
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


class Plotter(BasePlotter):
    def __init__(self, state: BaseState) -> None:
        self.ui = pn.Column()
        super().__init__(state)
        self.float_panels = pn.Column(sizing_mode="stretch_width")
        self.variable_selector = pn.widgets.Select(width=400)
        self.add_plot_button = pn.widgets.Button(
            name="Add Plot",
            button_type="primary",
            on_click=self._add_plot_callback,
        )
        self.close_all_button = pn.widgets.Button(
            name="Close All Plots",
            button_type="danger",
            on_click=self._close_all_plots_callback,
        )

        self.plotting_controls = pn.Row(
            self.variable_selector,
            self.add_plot_button,
            self.close_all_button,
            sizing_mode="stretch_width",
            align="center",
        )

        self.filter_input = pn.widgets.TextInput(
            placeholder="Filter...",
            width=400,
        )
        self.filter_input.param.watch(self._update_filter_view, "value_input")
        self.ui.extend(
            [self.filter_input, self.plotting_controls, self.float_panels]
        )

    def _update_filter_view(self, event):
        """Updates the variable selector based on the filter text."""
        filter_text = self.filter_input.value_input.lower()
        options = [
            full_path
            for full_path in self._state.variables
            if not filter_text or filter_text in full_path.lower()
        ]
        self.variable_selector.options = sorted(options)

    def get_dashboard(self):
        return self.ui

    @param.depends("_state.variables", watch=True)
    def _update_variable_selector(self) -> None:
        """Updates the variable selector when new variables are discovered."""
        self.variable_selector.options = sorted(
            list(self._state.variables.keys())
        )

    def _close_all_plots_callback(self, event) -> None:
        """Closes all active plot panels."""
        for float_panel in self.float_panels:
            float_panel.status = "closed"

    def _add_plot_callback(self, event) -> None:
        """Adds a new plot panel for the selected variable."""
        full_path = self.variable_selector.value
        if not full_path or full_path not in self._state.variables:
            return

        var = self._state.variables[full_path]
        if var.is_visualized:
            return  # Plot already exists

        var.is_visualized = True

        plot_func = functools.partial(
            self._plot_variable_vs_time, full_path=full_path
        )
        dynamic_plot = pn.pane.HoloViews(
            hv.DynamicMap(param.bind(plot_func, time=self.param.time)).opts(
                framewise=True, axiswise=True
            ),
            sizing_mode="stretch_both",
        )
        float_panel = ResizableFloatPanel(
            dynamic_plot,
            name=var.full_path,
            position="left-top",
            offsetx=random.randint(0, 2000),
            offsety=random.randint(0, 1000),
            contained=False,
        )

        def on_status_change(event):
            if event.new == "closed":
                self._floatpanel_closed_callback(full_path)

        float_panel.param.watch(on_status_change, "status")
        self.float_panels.append(float_panel)

    def _floatpanel_closed_callback(self, full_path: str, event=None) -> None:
        """Handles cleanup when a plot panel is closed."""
        if full_path in self._state.variables:
            var = self._state.variables[full_path]
            var.is_visualized = False
            self._state.data.pop(var.full_path, None)

    def plot_empty(self, name: str, var_dim: Dim):
        """Returns an empty plot to show when no data is available."""
        title = f"No data for t = {self.time}"
        if var_dim == Dim.TWO_D:
            return hv.QuadMesh(
                (np.array([0]), np.array([0]), np.zeros((1, 1))),
                kdims=["x", "y"],
                vdims=[name],
            ).opts(title=title, responsive=True)
        return hv.Curve(([], []), kdims=["time"], vdims=["value"]).opts(
            title=title, responsive=True
        )

    def plot_1d(self, ds: xr.Dataset, var: Variable, time_index: int):
        """Generates a 1D plot for a given time index."""
        data_var = ds[var.full_path].isel(time=time_index).values
        coord_name = var.coord_names[0]
        coord_var = (
            ds[f"{var.full_path}_{coord_name}"].isel(time=time_index).values
        )
        title = f"{var.full_path} (t={float(ds.time.values[time_index]):.3f}s)"
        return hv.Curve(
            (coord_var, data_var), kdims=[coord_name], vdims=[var.full_path]
        ).opts(title=title, responsive=True)

    def plot_2d(self, ds: xr.Dataset, var: Variable, time_index: int):
        """Generates a 2D plot for a given time index."""
        y_name, x_name = var.coord_names
        data_var = ds[var.full_path].isel(time=time_index).values
        x = ds[f"{var.full_path}_{x_name}"].isel(time=time_index).values
        y = ds[f"{var.full_path}_{y_name}"].isel(time=time_index).values
        title = f"{var.full_path} (t={float(ds.time.values[time_index]):.3f}s)"

        return hv.QuadMesh(
            (x, y, data_var),
            kdims=[x_name, y_name],
            vdims=[var.full_path],
        ).opts(
            cmap="viridis",
            colorbar=True,
            framewise=True,
            title=title,
            responsive=True,
            xlabel=x_name,
            ylabel=y_name,
        )

    def _plot_variable_vs_time(self, full_path: str, time: float):
        """Core plotting function that dispatches to specific plot types."""
        var = self.active_state.variables.get(full_path)
        if not var:
            return self.plot_empty("unknown", Dim.ZERO_D)

        ds = self.active_state.data.get(var.full_path)
        if ds is None or len(ds.time) == 0:
            return self.plot_empty(var.full_path, var.dimension)

        time_array = ds.time.values
        if time not in time_array:
            return self.plot_empty(var.full_path, var.dimension)

        time_index = np.where(time_array == time)[0][0]

        if var.dimension == Dim.ZERO_D:
            t_vals = time_array[: time_index + 1]
            v_vals = (
                ds[var.full_path].isel(time=slice(0, time_index + 1)).values
            )
            return hv.Curve(
                (t_vals, v_vals), kdims=["time"], vdims=[var.full_path]
            ).opts(title=f"{var.full_path} vs time", responsive=True)
        elif var.dimension == Dim.ONE_D:
            return self.plot_1d(ds, var, time_index)
        elif var.dimension == Dim.TWO_D:
            return self.plot_2d(ds, var, time_index)

    def __panel__(self) -> Viewable:
        return self._panel

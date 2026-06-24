"""Tests for the distilled-store viewer (data layer, plotting, plugin hook)."""

import holoviews as hv
import numpy as np
import xarray as xr

from imas_muscle3.distill import Distiller, ZarrSink
from imas_muscle3.viewer import store as store_mod
from imas_muscle3.viewer.panel_app import RunPanel, make_panel
from imas_muscle3.viewer.plots import plot_overlay, plot_variable

# --- fixtures: a realistic distilled store written like the actor would -----


def _write_equilibrium_store(run_dir, occurrence="0000"):
    """Distill equilibrium slices into <run>/.../equilibrium_in/<occ>.zarr."""
    import imas

    workdir = run_dir / "instances" / "eq" / "workdir"
    store = workdir / "equilibrium_in" / f"{occurrence}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    sink = ZarrSink(store)
    distiller = Distiller(auto=True)
    for t, ip in [(0.0, 1e6), (1.0, 1.1e6), (2.0, 1.2e6)]:
        eq = imas.IDSFactory("4.0.0").equilibrium()
        eq.ids_properties.homogeneous_time = 1
        eq.time = [t]
        eq.time_slice.resize(1)
        eq.time_slice[0].global_quantities.ip = ip
        eq.time_slice[0].profiles_1d.psi = np.linspace(0, 1, 8)
        eq.time_slice[0].profiles_1d.f_df_dpsi = np.ones(8) * ip
        for name, ds in distiller.distill(eq).items():
            sink.append(name, ds)
    sink.close()
    return store


# --- data layer -------------------------------------------------------------


def test_find_stores_and_groups(tmp_path):
    store = _write_equilibrium_store(tmp_path)
    found = store_mod.find_stores(tmp_path)
    assert found == [store]
    assert store_mod.list_groups(store) == ["equilibrium"]
    assert store_mod.store_port(store) == "equilibrium_in"
    # The fixture writes under instances/eq/workdir, so the label is scoped by
    # the instance name (disambiguating recorders that share a port name).
    assert store_mod.store_instance(store) == "eq"
    assert store_mod.store_label(store) == "eq/equilibrium_in/0000"


def test_occurrences_group_by_reuse(tmp_path):
    _write_equilibrium_store(tmp_path, occurrence="0000")
    _write_equilibrium_store(tmp_path, occurrence="0001")
    grouped = store_mod.occurrences(tmp_path)
    assert list(grouped) == ["0000", "0001"]
    assert set(grouped["0000"]) == {"equilibrium_in"}
    assert grouped["0001"]["equilibrium_in"].stem == "0001"


def test_plottable_variables_and_rank(tmp_path):
    store = _write_equilibrium_store(tmp_path)
    ds = store_mod.open_group(store, "equilibrium")
    variables = store_mod.plottable_variables(ds)
    assert "time_slice.global_quantities.ip" in variables
    assert "time_slice.profiles_1d.f_df_dpsi" in variables
    assert store_mod.variable_rank(ds["time_slice.global_quantities.ip"]) == 0
    assert store_mod.variable_rank(ds["time_slice.profiles_1d.f_df_dpsi"]) == 1
    # The profile carries its DD coordinate.
    assert store_mod.coord_names(ds["time_slice.profiles_1d.f_df_dpsi"]) == [
        "time_slice.profiles_1d.psi"
    ]


# --- plotting dispatch ------------------------------------------------------


def _ds_0d():
    return xr.Dataset(
        {"v": ("time", [1.0, 2.0, 3.0])},
        coords={"time": [0.0, 1.0, 2.0]},
    )


def _ds_1d():
    x = np.arange(5.0)
    ds = xr.Dataset(
        {"v": (("time", "x"), [np.ones(5), np.full(5, 2.0)])},
        coords={"time": [0.0, 1.0], "psi": (("time", "x"), [x, x])},
    )
    ds["v"].attrs.update(coordinates="time psi", units="A")
    return ds


def _ds_2d():
    z = np.arange(6.0).reshape(2, 3)
    return xr.Dataset(
        {"v": (("time", "y", "x"), [z])},
        coords={"time": [0.0]},
    )


def test_plot_0d_is_time_curve():
    el = plot_variable(_ds_0d(), "v", time_index=2)
    assert isinstance(el, hv.Curve)
    assert len(el) == 3  # full time series


def test_plot_1d_is_profile_at_time():
    el = plot_variable(_ds_1d(), "v", time_index=1)
    assert isinstance(el, hv.Curve)
    assert list(el.dimension_values("v")) == [2.0] * 5


def test_plot_2d_is_quadmesh():
    el = plot_variable(_ds_2d(), "v", time_index=0)
    assert isinstance(el, hv.QuadMesh)


def test_overlay_single_rank1_falls_through():
    # A single profile (rank 1) is just its plot_variable curve, no marker.
    el = plot_overlay(_ds_1d(), ["v"], time_index=1)
    assert isinstance(el, hv.Curve)


def test_overlay_rank0_has_time_marker():
    # Over-time plots get a dashed playhead at the current time.
    el = plot_overlay(_ds_0d(), ["v"], time_index=2)
    assert isinstance(el, hv.Overlay)
    assert any(isinstance(e, hv.VLine) for e in el)


def test_overlay_multiple_variables_is_overlay():
    ds = xr.Dataset(
        {
            "a": ("time", [1.0, 2.0]),
            "b": ("time", [3.0, 4.0]),
        },
        coords={"time": [0.0, 1.0]},
    )
    el = plot_overlay(ds, ["a", "b"], time_index=1)
    assert isinstance(el, hv.Overlay)
    # two curves plus the time-marker playhead
    assert sum(isinstance(e, hv.Curve) for e in el) == 2
    assert any(isinstance(e, hv.VLine) for e in el)


def test_plot_clamps_time_index():
    # Out-of-range index must not raise.
    el = plot_variable(_ds_1d(), "v", time_index=99)
    assert isinstance(el, hv.Curve)


# --- plugin hook ------------------------------------------------------------


def test_make_panel_none_without_stores(tmp_path):
    assert make_panel(tmp_path) is None


def test_make_panel_returns_panel_with_stores(tmp_path):
    _write_equilibrium_store(tmp_path)
    panel = make_panel(tmp_path)
    assert isinstance(panel, RunPanel)
    assert panel.title == "IMAS plots"
    # The view builds without a live session.
    assert panel.view() is not None

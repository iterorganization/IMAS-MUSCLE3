import imas
import numpy as np
import pytest
import xarray as xr

from imas_muscle3.distill import Distiller, ZarrSink, group_name


# --- distiller ------------------------------------------------------------
#
# The actor receives single time slices, so these build single-slice IDSs
# (the distiller canonicalizes them to one homogeneous time point).


def _cp_slice(t, density):
    cp = imas.IDSFactory("4.0.0").core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.time = [t]
    cp.profiles_1d.resize(1)
    cp.profiles_1d[0].grid.rho_tor_norm = np.linspace(0.0, 1.0, len(density))
    cp.profiles_1d[0].electrons.density = np.asarray(density, dtype=float)
    return cp


def _eq_slice(t, ip):
    eq = imas.IDSFactory("4.0.0").equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = [t]
    eq.time_slice.resize(1)
    eq.time_slice[0].global_quantities.ip = ip
    return eq


def test_distiller_discovers_profiles():
    # Quantities are tensorized into one dataset keyed by the IDS name, with
    # imas-python's netCDF naming (dotted paths, real coords, units).
    out = Distiller(auto=True).distill(_cp_slice(0.5, np.ones(16)))
    assert "core_profiles" in out
    ds = out["core_profiles"]
    density = "profiles_1d.electrons.density"
    assert density in ds.data_vars
    assert ds.sizes["time"] == 1
    assert ds[density].attrs["units"] == "m^-3"
    assert "profiles_1d.grid.rho_tor_norm" in ds.coords
    assert list(ds[density].values[0]) == list(np.ones(16))


def test_distiller_discovers_scalar():
    out = Distiller(auto=True).distill(_eq_slice(0.0, 1e6))
    assert "equilibrium" in out
    ip = out["equilibrium"]["time_slice.global_quantities.ip"]
    assert ip.dims == ("time",)
    assert float(ip.values[0]) == pytest.approx(1e6)


def test_normalize_time_combines_axes():
    # A message can carry several coincident time axes (equilibrium's
    # time_slice.time + grids_ggd.time); they collapse onto one 'time'.
    from imas_muscle3.distill.distiller import _normalize_time

    ds = xr.Dataset(
        {
            "ip": ("time_slice.time", [1.0]),
            "psi": (("time_slice.time", "x"), np.ones((1, 3))),
            "b0": ("grids_ggd.time", [7.0]),
        },
        coords={"time_slice.time": [2.5], "grids_ggd.time": [2.5]},
    )
    out = _normalize_time(ds)
    assert set(out.dims) >= {"time", "x"}
    assert "time_slice.time" not in out.dims
    assert "grids_ggd.time" not in out.dims
    for v in ("ip", "psi", "b0"):
        assert "time" in out[v].dims
    assert list(out["time"].values) == [2.5]


def test_distiller_accepts_whole_trace():
    # A multi-slice IDS is tensorized as one dataset with a full time axis.
    eq = imas.IDSFactory("4.0.0").equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = [0.0, 1.0, 2.0]
    eq.time_slice.resize(3)
    for i, t in enumerate(eq.time):
        eq.time_slice[i].global_quantities.ip = 1e6 + i * 1e5
    out = Distiller(auto=True).distill(eq)
    ip = out["equilibrium"]["time_slice.global_quantities.ip"]
    assert ip.dims == ("time",)
    assert list(ip.values) == pytest.approx([1e6, 1.1e6, 1.2e6])


def test_distiller_needs_auto_or_config():
    with pytest.raises(ValueError):
        Distiller(auto=False)


def test_distiller_config_callable(equilibrium):
    def extract(ids):
        return {
            "custom/marker": xr.Dataset(
                {"value": ("time", [42.0])},
                coords={"time": [float(ids.time[0])]},
            )
        }

    out = Distiller(auto=False, extract=extract).distill(equilibrium)
    assert set(out) == {"custom/marker"}
    assert float(out["custom/marker"]["value"].values[0]) == 42.0


# --- zarr sink ------------------------------------------------------------


def test_group_name_is_slash_free():
    assert group_name("equilibrium/time_slice[0]/global_quantities/ip") == (
        "equilibrium.time_slice[0].global_quantities.ip"
    )


def _single_1d(t, values):
    arr = np.asarray(values, dtype=float)
    return xr.Dataset(
        {
            "value": (("time", "dim0"), arr[np.newaxis, :]),
            "coord0": (
                ("time", "dim0"),
                np.arange(len(arr), dtype=float)[np.newaxis, :],
            ),
        },
        coords={"time": [t]},
        attrs={"full_path": "x/y"},
    )


def test_zarr_sink_combines_along_time(tmp_path):
    store = tmp_path / "core_profiles_in.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))
    sink.append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert list(ds.time.values) == [0.0, 1.0]
    assert ds["value"].shape == (2, 8)
    assert ds.attrs["full_path"] == "x/y"


def test_zarr_sink_writes_whole_trace(tmp_path):
    # A single message carrying a whole trace (time>1) is written in one go.
    store = tmp_path / "equilibrium_in.zarr"
    ds = xr.Dataset(
        {"value": (("time", "dim0"), np.ones((49, 8)))},
        coords={"time": np.arange(49.0)},
        attrs={"full_path": "x/y"},
    )
    sink = ZarrSink(store)
    sink.append("x/y", ds)
    sink.close()
    out = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert out["value"].shape == (49, 8)


def test_zarr_sink_pads_ragged_profiles(tmp_path):
    store = tmp_path / "store.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))
    # A shorter later slice is NaN-padded up to the max width.
    sink.append("x/y", _single_1d(1.0, np.full(5, 3.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert ds["value"].shape == (2, 8)
    second = ds["value"].values[1]
    assert list(second[:5]) == [3.0] * 5
    assert np.isnan(second[5:]).all()


def test_zarr_sink_combines_gaps(tmp_path):
    # The TORAX case: some messages carry a var, others don't. The store must
    # still open, union time axis, NaN where the var is absent.
    store = tmp_path / "core_profiles_in.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))  # has 'value'
    sink.append(
        "x/y",
        xr.Dataset({"other": ("time", [9.0])}, coords={"time": [0.5]}),
    )  # gap: no 'value'
    sink.append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert list(ds.time.values) == [0.0, 0.5, 1.0]
    assert ds["value"].shape == (3, 8)
    assert np.isnan(ds["value"].values[1]).all()  # gap NaN-filled
    assert np.isnan(ds["other"].values[0]) and ds["other"].values[1] == 9.0

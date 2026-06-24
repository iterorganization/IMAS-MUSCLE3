import multiprocessing

import imas
import numpy as np
import pytest
import xarray as xr
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.distill import Distiller, ZarrSink, group_name
from tests.ymmsl_helpers import load_config

"""Force 'spawn' start method to avoid deadlocks with pytest."""
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


# --- distiller unit tests -------------------------------------------------
#
# The actor receives single time slices, so the unit tests build single-slice
# IDSs (the distiller canonicalizes them to one homogeneous time point).


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
    # imas-python's netCDF naming conventions (dotted paths, real coords, units).
    out = Distiller(auto=True).distill(_cp_slice(0.5, np.ones(16)))
    assert "core_profiles" in out
    ds = out["core_profiles"]
    density = "profiles_1d.electrons.density"
    assert density in ds.data_vars
    assert ds.sizes["time"] == 1
    assert ds[density].attrs["units"] == "m^-3"
    # The profile's coordinate is a proper dataset coordinate.
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
    # time_slice.time + grids_ggd.time; core_profiles' root time + profiles_1d).
    # They must collapse onto one 'time' so every quantity shares it.
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
    # A multi-slice IDS is tensorized as one dataset with a full time axis,
    # and the time-like dim is normalized to 'time'.
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
                attrs={"dimension": "0D"},
            )
        }

    out = Distiller(auto=False, extract=extract).distill(equilibrium)
    assert set(out) == {"custom/marker"}
    assert float(out["custom/marker"]["value"].values[0]) == 42.0


# --- zarr sink unit tests -------------------------------------------------


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
        attrs={"dimension": "1D", "full_path": "x/y"},
    )


def test_zarr_sink_combines_along_time(tmp_path):
    store = tmp_path / "core_profiles_in.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))
    sink.append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    sink.close()  # nothing is written until close

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
    # The TORAX out_i case: some messages carry a var, others don't. The store
    # must still open, with the union time axis and NaN where the var is absent.
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


# --- integration test: two timelines -> two zarr stores -------------------


def _ymmsl_two_timelines(eq_uri, cp_uri, store_path):
    return f"""
ymmsl_version: v0.1
model:
  name: test_distill
  components:
    eq_source:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    cp_source:
      implementation: source_component
      ports:
        o_i: [core_profiles_out]
    distill:
      implementation: distill_component
      ports:
        s: [equilibrium_in, core_profiles_in]
  conduits:
    eq_source.equilibrium_out: distill.equilibrium_in
    cp_source.core_profiles_out: distill.core_profiles_in
settings:
  eq_source.source_uri: {eq_uri}
  cp_source.source_uri: {cp_uri}
  distill.store_path: {store_path}
  distill.monitor_interval: 0.05
implementations:
  distill_component:
    executable: python
    args: -u -m imas_muscle3.actors.distill_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  eq_source:
    threads: 1
  cp_source:
    threads: 1
  distill:
    threads: 1
"""


def test_distill_records_two_timelines(tmp_path, equilibrium, core_profiles):
    eq_path = (tmp_path / "eq_source_data").absolute()
    cp_path = (tmp_path / "cp_source_data").absolute()
    eq_uri = f"imas:hdf5?path={eq_path}"
    cp_uri = f"imas:hdf5?path={cp_path}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(equilibrium)
    with DBEntry(cp_uri, "w") as entry:
        entry.put(core_profiles)

    store_path = (tmp_path / "distill_store").absolute()
    config = load_config(_ymmsl_two_timelines(eq_uri, cp_uri, store_path))
    run_dir = RunDir(tmp_path / "run")
    manager = Manager(config, run_dir)
    manager.start_instances()
    assert manager.wait()

    # One store per timeline per reuse: <port>/<occurrence>.zarr. The source
    # streams the whole trace in a single reuse, so occurrence 0000.
    eq_store = store_path / "equilibrium_in" / "0000.zarr"
    cp_store = store_path / "core_profiles_in" / "0000.zarr"
    assert eq_store.is_dir()
    assert cp_store.is_dir()

    # Each timeline's quantities live in one group named after the IDS, with
    # imas-python conventions (dotted var names, real coords). The plasma
    # current grew to all three time steps.
    eq = xr.open_zarr(eq_store, group="equilibrium", consolidated=False)
    ip = eq["time_slice.global_quantities.ip"]
    assert list(eq.time.values) == list(equilibrium.time)
    assert ip.dims == ("time",)
    assert float(ip.values[0]) == pytest.approx(1e6)

    # An electron-density profile was distilled with its rho_tor_norm coord.
    cp = xr.open_zarr(cp_store, group="core_profiles", consolidated=False)
    density = cp["profiles_1d.electrons.density"]
    assert list(cp.time.values) == list(core_profiles.time)
    assert density.shape == (len(core_profiles.time), 16)
    assert "profiles_1d.grid.rho_tor_norm" in cp.coords


# --- occurrence labelling from the message stream -------------------------


def _msg(t, nxt, data):
    from libmuscle import Message

    return Message(t, nxt, data=data)


def test_distill_handler_splits_occurrences_on_restart(tmp_path, equilibrium):
    """A backward time step / end-of-stream starts a new occurrence."""
    from imas_muscle3.actors.distill_component import DistillHandler

    data = equilibrium.serialize()
    store_dir = tmp_path / "equilibrium_in"
    handler = DistillHandler(store_dir, "equilibrium", Distiller(auto=True))
    # iteration 0: t=0,1 (1 ends the stream); iteration 1: t=0,1 (time resets).
    handler.handle(0, _msg(0.0, 1.0, data))
    handler.handle(1, _msg(1.0, None, data))
    handler.handle(2, _msg(0.0, 1.0, data))
    handler.handle(3, _msg(1.0, None, data))
    handler.close()

    assert (store_dir / "0000.zarr").is_dir()
    assert (store_dir / "0001.zarr").is_dir()
    assert not (store_dir / "0002.zarr").exists()


def test_distill_handler_one_occurrence_for_monotonic(tmp_path, equilibrium):
    """A single monotonic trace stays one occurrence (no spurious split)."""
    from imas_muscle3.actors.distill_component import DistillHandler

    data = equilibrium.serialize()
    store_dir = tmp_path / "equilibrium_in"
    handler = DistillHandler(store_dir, "equilibrium", Distiller(auto=True))
    handler.handle(0, _msg(0.0, 1.0, data))
    handler.handle(1, _msg(1.0, 2.0, data))
    handler.handle(2, _msg(2.0, None, data))
    handler.close()

    assert (store_dir / "0000.zarr").is_dir()
    assert not (store_dir / "0001.zarr").exists()

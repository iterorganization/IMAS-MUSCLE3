"""Tests for the visualization-profile mechanism and the reference profile."""

import multiprocessing

import imas
import numpy as np
import panel as pn
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.distill import Distiller, ZarrSink
from imas_muscle3.distill.profiles import equilibrium_overview as profile
from imas_muscle3.viewer import store as store_mod
from imas_muscle3.viewer.profile import ProfileData, load_profile
from tests.ymmsl_helpers import load_config

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

PROFILE_PATH = profile.__file__


def _equilibrium(times):
    eq = imas.IDSFactory("4.0.0").equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = list(times)
    eq.time_slice.resize(len(times))
    theta = np.linspace(0, 2 * np.pi, 40)
    for i, t in enumerate(times):
        ts = eq.time_slice[i]
        ts.boundary.outline.r = 6 + 1.5 * np.cos(theta)
        ts.boundary.outline.z = 2.2 * np.sin(theta)
        ts.global_quantities.ip = 1e6 + i * 1e5
        ts.global_quantities.beta_tor = 0.1 + i * 0.01
        ts.profiles_1d.psi = np.linspace(0, 1, 8)
        ts.profiles_1d.f_df_dpsi = np.ones(8) * (i + 1)
        ts.profiles_1d.dpressure_dpsi = np.linspace(0, 2, 8)
    return eq


def _write_profile_store(workdir, times=(0.0, 1.0, 2.0)):
    """Distill via the profile's extract into a store, as the actor would."""
    workdir.mkdir(parents=True, exist_ok=True)
    sink = ZarrSink(workdir / "equilibrium_in.zarr")
    distiller = Distiller(auto=False, extract=profile.extract)
    for t in times:
        eq = _equilibrium([t])
        for name, ds in distiller.distill(eq, time=t).items():
            sink.append(name, ds)
    return workdir / "equilibrium_in.zarr"


# --- the reference profile's extract ---------------------------------------


def test_extract_produces_derived_groups():
    out = profile.extract(_equilibrium([0.5]))
    assert set(out) == {
        "derived/separatrix",
        "derived/global",
        "derived/profiles_1d",
    }
    assert out["derived/global"]["ip"].dims == ("time",)
    assert out["derived/separatrix"].attrs["kind"] == "curve_rz"


# --- ProfileData + load_profile + plot -------------------------------------


def test_load_profile_exposes_extract_and_plot():
    loaded = load_profile(PROFILE_PATH)
    assert callable(loaded.extract) and callable(loaded.plot)


def test_profile_data_reads_by_logical_name(tmp_path):
    store = _write_profile_store(tmp_path / "instances" / "eq" / "workdir")
    data = ProfileData({store.stem: store})
    # The logical key used when distilling resolves to the dotted on-disk group.
    sep = data.dataset("derived/separatrix")
    assert sep is not None
    assert list(sep["r"].dims) == ["time", "pt"]
    assert data.n_times() == 3


def test_profile_plot_builds_a_panel(tmp_path):
    store = _write_profile_store(tmp_path / "instances" / "eq" / "workdir")
    data = ProfileData({store.stem: store})
    view = profile.plot(data, time_index=1)
    assert isinstance(view, pn.Row)


# --- actor config setting, end to end --------------------------------------


def _ymmsl(eq_uri, store_path, config):
    return f"""
ymmsl_version: v0.1
model:
  name: test_distill_profile
  components:
    eq_source:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    distill:
      implementation: distill_component
      ports:
        s: [equilibrium_in]
  conduits:
    eq_source.equilibrium_out: distill.equilibrium_in
settings:
  eq_source.source_uri: {eq_uri}
  distill.store_path: {store_path}
  distill.auto: false
  distill.config: {config}
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
  distill:
    threads: 1
"""


def test_actor_uses_profile_config(tmp_path):
    eq_path = (tmp_path / "eq_data").absolute()
    eq_uri = f"imas:hdf5?path={eq_path}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(_equilibrium([0.0, 1.0, 2.0]))

    store_path = (tmp_path / "store").absolute()
    config = load_config(_ymmsl(eq_uri, store_path, PROFILE_PATH))
    manager = Manager(config, RunDir(tmp_path / "run"))
    manager.start_instances()
    assert manager.wait()

    store = store_path / "equilibrium_in.zarr"
    # The profile's derived groups were written (auto was off).
    assert "derived.separatrix" in store_mod.list_groups(store)
    assert "derived.global" in store_mod.list_groups(store)
    # And the store records which profile produced it, so the viewer finds it.
    assert store_mod.store_profile(store) == PROFILE_PATH

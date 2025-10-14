from pathlib import Path

import pytest
import ymmsl
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir


def test_source_to_sink(tmpdir, core_profiles):
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    data_sink_path = (Path(tmpdir) / "sink_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)
    tmppath = Path(str(tmpdir))
    # make config
    ymmsl_text = f"""
ymmsl_version: v0.1
model:
  name: test_model
  components:
    source_component:
      implementation: source_component
      ports:
        o_i: [core_profiles_out]
    sink_component:
      implementation: sink_component
      ports:
        f_init: [core_profiles_in]
  conduits:
    source_component.core_profiles_out: sink_component.core_profiles_in
settings:
  source_component.source_uri: {source_uri}
  sink_component.sink_uri: {sink_uri}
implementations:
  sink_component:
    executable: python
    args: -u -m imas_muscle3.actors.sink_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  source_component:
    threads: 1
  sink_component:
    threads: 1
"""

    config = ymmsl.load(ymmsl_text)

    # set up
    run_dir = RunDir(tmppath / "run")

    # launch MUSCLE Manager with simulation
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    # check that all went well
    assert success

    assert data_sink_path.exists()
    with DBEntry(sink_uri, "r") as entry:
        assert all(entry.get("core_profiles").time == core_profiles.time)


@pytest.mark.parametrize('use_sink', [True, False])
def test_source_to_hybrid_to_sink(tmpdir, core_profiles, use_sink):
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    data_sink_path = (Path(tmpdir) / "sink_component_data").absolute()
    data_hybrid_source_path = (Path(tmpdir) / "source_hybrid_component_data").absolute()
    data_hybrid_sink_path = (Path(tmpdir) / "sink_hybrid_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    hybrid_source_uri = f"imas:hdf5?path={data_hybrid_source_path}"
    hybrid_sink_uri = f"imas:hdf5?path={data_hybrid_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)
    with DBEntry(hybrid_source_uri, "w") as entry:
        entry.put(core_profiles)
    tmppath = Path(str(tmpdir))
    # make config
    ymmsl_text = f"""
    ymmsl_version: v0.1
    model:
      name: test_model
      components:
        source_component:
          implementation: source_component
          ports:
            o_i: [core_profiles_out]
        sink_component:
          implementation: sink_component
          ports:
            f_init: [core_profiles_in]
        hybrid_component:
          implementation: hybrid_component
          ports:
            f_init: [core_profiles_in]
            o_f: [core_profiles_out]
      conduits:
        source_component.core_profiles_out: hybrid_component.core_profiles_in
        hybrid_component.core_profiles_out: sink_component.core_profiles_in
    settings:
      source_component.source_uri: {source_uri}
      sink_component.sink_uri: {sink_uri}
      hybrid_component.source_uri: {hybrid_source_uri}
      {f"hybrid_component.sink_uri: {hybrid_sink_uri}" if use_sink else ''}
    implementations:
      sink_component:
        executable: python
        args: -u -m imas_muscle3.actors.sink_component
      source_component:
        executable: python
        args: -u -m imas_muscle3.actors.source_component
      hybrid_component:
        executable: python
        args: -u -m imas_muscle3.actors.sink_source_component
    resources:
      source_component:
        threads: 1
      sink_component:
        threads: 1
      hybrid_component:
        threads: 1
    """

    config = ymmsl.load(ymmsl_text)

    # set up
    run_dir = RunDir(tmppath / "run")

    # launch MUSCLE Manager with simulation
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    # check that all went well
    assert success

    assert data_sink_path.exists()
    with DBEntry(sink_uri, "r") as entry:
        assert all(entry.get("core_profiles").time == core_profiles.time)
    if use_sink:
        assert data_hybrid_sink_path.exists()
        with DBEntry(hybrid_sink_uri, "r") as entry:
            assert all(entry.get("core_profiles").time == core_profiles.time)
    else:
        assert not data_hybrid_sink_path.exists()


def test_source_with_time_range(tmpdir, core_profiles):
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    data_sink_path = (Path(tmpdir) / "sink_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)
    tmppath = Path(str(tmpdir))
    # make config
    ymmsl_text = f"""
ymmsl_version: v0.1
model:
  name: test_model
  components:
    source_component:
      implementation: source_component
      ports:
        o_i: [core_profiles_out]
    sink_component:
      implementation: sink_component
      ports:
        f_init: [core_profiles_in]
  conduits:
    source_component.core_profiles_out: sink_component.core_profiles_in
settings:
  source_component.source_uri: {source_uri}
  source_component.t_min: 0.5
  source_component.t_max: 1.5
  sink_component.sink_uri: {sink_uri}
implementations:
  sink_component:
    executable: python
    args: -u -m imas_muscle3.actors.sink_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  source_component:
    threads: 1
  sink_component:
    threads: 1
"""

    config = ymmsl.load(ymmsl_text)

    # set up
    run_dir = RunDir(tmppath / "run")

    # launch MUSCLE Manager with simulation
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    # check that all went well
    assert success

    assert data_sink_path.exists()
    with DBEntry(sink_uri, "r") as entry:
        assert all(core_profiles.time == [0, 1, 2])
        assert all(entry.get("core_profiles").time == [1])


def test_source_without_time_array(tmpdir, iron_core, pf_active):
    """
    Test if t_array in source is taken from pf_active even if
    iron_core is first in list
    """
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    data_sink_path = (Path(tmpdir) / "sink_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(iron_core)
        entry.put(pf_active)
    tmppath = Path(str(tmpdir))
    # make config
    ymmsl_text = f"""
ymmsl_version: v0.1
model:
  name: test_model
  components:
    source_component:
      implementation: source_component
      ports:
        o_i: [iron_core_out, pf_active_out]
    sink_component:
      implementation: sink_component
      ports:
        f_init: [iron_core_in, pf_active_in]
  conduits:
    source_component.iron_core_out: sink_component.iron_core_in
    source_component.pf_active_out: sink_component.pf_active_in
settings:
  source_component.source_uri: {source_uri}
  sink_component.sink_uri: {sink_uri}
implementations:
  sink_component:
    executable: python
    args: -u -m imas_muscle3.actors.sink_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  source_component:
    threads: 1
  sink_component:
    threads: 1
"""

    config = ymmsl.load(ymmsl_text)

    # set up
    run_dir = RunDir(tmppath / "run")

    # launch MUSCLE Manager with simulation
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    # check that all went well
    assert success

    assert data_sink_path.exists()
    with DBEntry(sink_uri, "r") as entry:
        assert all(pf_active.time == [0, 1, 2])
        assert all(entry.get("pf_active").time == [0, 1, 2])

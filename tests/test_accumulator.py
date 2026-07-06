from pathlib import Path

import pytest
import ymmsl
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir


@pytest.mark.parametrize("use_t_next", [True, False])
def test_accumulator(tmpdir, core_profiles, use_t_next):
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    data_sink_path = (Path(tmpdir) / "sink_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)
    tmppath = Path(str(tmpdir))
    # whether or not optional override port is used for t_next
    if use_t_next:
        ports = "[core_profiles_in, t_next]"
        conduit = "source.core_profiles_out: accumulator.t_next"
    else:
        ports = "[core_profiles_in]"
        conduit = ""
    # make config
    ymmsl_text = f"""
ymmsl_version: v0.2
models:
  test_model:
    components:
      source:
        description: source component
        implementation: source
        ports:
          o_i: [core_profiles_out]
      accumulator:
        description: accumulator component
        implementation: accumulator
        ports:
          s: {ports}
          o_f: [core_profiles_out]
      sink:
        description: sink component
        implementation: sink
        ports:
          f_init: [core_profiles_in]
    conduits:
      source.core_profiles_out: accumulator.core_profiles_in
      {conduit}
      accumulator.core_profiles_out: sink.core_profiles_in
settings:
  source.source_uri: {source_uri}
  sink.sink_uri: {sink_uri}
programs:
  sink:
    executable: python
    args: -u -m imas_muscle3.actors.sink_component
  source:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
  accumulator:
    executable: python
    args: -u -m imas_muscle3.actors.accumulator_component
resources:
  sink:
    threads: 1
  source:
    threads: 1
  accumulator:
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

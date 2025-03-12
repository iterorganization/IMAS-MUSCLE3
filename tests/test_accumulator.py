from pathlib import Path

import ymmsl
from imaspy import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir


def test_accumulator(tmpdir, core_profiles):
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
    accumulator_component:
      implementation: accumulator_component
      ports:
        s: [core_profiles_in, core_profiles_beep]
        o_f: [core_profiles_out]
    sink_component:
      implementation: sink_component
      ports:
        f_init: [core_profiles_in]
  conduits:
    source_component.core_profiles_out: accumulator_component.core_profiles_in
    source_component.core_profiles_out: accumulator_component.core_profiles_beep
    accumulator_component.core_profiles_out: sink_component.core_profiles_in
settings:
  source_component.source_uri: {source_uri}
  sink_component.sink_uri: {sink_uri}
implementations:
  sink_component:
    executable: python
    args: -u -m imas_m3.actors.sink_component
  source_component:
    executable: python
    args: -u -m imas_m3.actors.source_component
  accumulator_component:
    executable: python
    args: -u -m imas_m3.actors.accumulator_component
resources:
  source_component:
    threads: 1
  sink_component:
    threads: 1
  accumulator_component:
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
        print(core_profiles.time)
        print(entry.get("core_profiles").time)
        assert all(entry.get("core_profiles").time == core_profiles.time)

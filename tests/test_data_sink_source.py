from pathlib import Path

import pytest
import ymmsl
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir
from libmuscle.pytest import MuscleTester

from conftest import deserialize, receive_all, slice_messages


def test_source_sends_all_slices(
    muscle3_tester: MuscleTester, tmp_path: Path, core_profiles
) -> None:
    source_uri = f"imas:hdf5?path={(tmp_path / 'source_data').absolute()}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)

    config = f"""
ymmsl_version: v0.2
programs:
  source_component:
    ports:
      o_i: [core_profiles_out]
    executable: python
    args: -u -m imas_muscle3.actors.source_component
settings:
  source_component.source_uri: {source_uri}
"""
    tester = muscle3_tester.start_implementation(config, "source_component")

    messages = receive_all(tester, "core_profiles_out")
    received_times = [msg.timestamp for msg in messages]
    assert received_times == list(core_profiles.time)
    for msg in messages:
        result = deserialize("core_profiles", msg.data)
        assert list(result.time) == [msg.timestamp]


@pytest.mark.parametrize("use_sink", [True, False])
def test_hybrid_component(
    muscle3_tester: MuscleTester, tmp_path: Path, core_profiles, use_sink: bool
) -> None:
    hybrid_source_uri = (
        f"imas:hdf5?path={(tmp_path / 'hybrid_source_data').absolute()}"
    )
    hybrid_sink_path = (tmp_path / "hybrid_sink_data").absolute()
    hybrid_sink_uri = f"imas:hdf5?path={hybrid_sink_path}"
    with DBEntry(hybrid_source_uri, "w") as entry:
        entry.put(core_profiles)

    sink_setting = (
        f"  hybrid_component.sink_uri: {hybrid_sink_uri}" if use_sink else ""
    )
    config = f"""
ymmsl_version: v0.2
programs:
  hybrid_component:
    ports:
      f_init: [core_profiles_in]
      o_f: [core_profiles_out]
    executable: python
    args: -u -m imas_muscle3.actors.sink_source_component
settings:
  hybrid_component.source_uri: {hybrid_source_uri}
{sink_setting}
"""
    tester = muscle3_tester.start_implementation(config, "hybrid_component")

    for msg in slice_messages(core_profiles, "core_profiles"):
        tester.send("core_profiles_in", msg)
        reply = tester.receive("core_profiles_out")
        result = deserialize("core_profiles", reply.data)
        # hybrid's own source_uri drives the O_F payload, at the timestamp
        # dictated by what came in on F_INIT
        assert result.time[0] == msg.timestamp

    # Shut the implementation down so its sink DBEntry is flushed and closed
    # before we read it back.
    muscle3_tester.cleanup()

    if use_sink:
        assert hybrid_sink_path.exists()
        with DBEntry(hybrid_sink_uri, "r") as entry:
            assert all(entry.get("core_profiles").time == core_profiles.time)
    else:
        assert not hybrid_sink_path.exists()


def test_source_with_time_range(
    muscle3_tester: MuscleTester, tmp_path: Path, core_profiles
) -> None:
    source_uri = f"imas:hdf5?path={(tmp_path / 'source_data').absolute()}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)

    config = f"""
ymmsl_version: v0.2
programs:
  source_component:
    ports:
      o_i: [core_profiles_out]
    executable: python
    args: -u -m imas_muscle3.actors.source_component
settings:
  source_component.source_uri: {source_uri}
  source_component.t_min: 0.5
  source_component.t_max: 1.5
"""
    tester = muscle3_tester.start_implementation(config, "source_component")

    assert all(core_profiles.time == [0, 1, 2])
    messages = receive_all(tester, "core_profiles_out")
    assert [msg.timestamp for msg in messages] == [1]
    result = deserialize("core_profiles", messages[0].data)
    assert list(result.time) == [1]


def test_non_iterative_source_with_time_range(
    muscle3_tester: MuscleTester, tmp_path: Path, core_profiles
) -> None:
    source_uri = f"imas:hdf5?path={(tmp_path / 'source_data').absolute()}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(core_profiles)

    config = f"""
ymmsl_version: v0.2
programs:
  source_component:
    ports:
      o_i: [core_profiles_out]
    executable: python
    args: -u -m imas_muscle3.actors.source_component
settings:
  source_component.source_uri: {source_uri}
  source_component.iterative: false
  source_component.t_min: 0.5
  source_component.t_max: 2.5
"""
    tester = muscle3_tester.start_implementation(config, "source_component")

    assert all(core_profiles.time == [0, 1, 2])
    messages = receive_all(tester, "core_profiles_out")
    assert len(messages) == 1
    result = deserialize("core_profiles", messages[0].data)
    assert list(result.time) == [1, 2]


def test_source_without_time_array(
    muscle3_tester: MuscleTester, tmp_path: Path, iron_core, pf_active
) -> None:
    """
    Test if t_array in source is taken from pf_active even if
    iron_core is first in list
    """
    source_uri = f"imas:hdf5?path={(tmp_path / 'source_data').absolute()}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(iron_core)
        entry.put(pf_active)

    config = f"""
ymmsl_version: v0.2
programs:
  source_component:
    ports:
      o_i: [iron_core_out, pf_active_out]
    executable: python
    args: -u -m imas_muscle3.actors.source_component
settings:
  source_component.source_uri: {source_uri}
"""
    tester = muscle3_tester.start_implementation(config, "source_component")

    assert all(pf_active.time == [0, 1, 2])
    pf_messages = receive_all(tester, "pf_active_out")
    receive_all(tester, "iron_core_out")
    assert [msg.timestamp for msg in pf_messages] == list(pf_active.time)


def ls_snapshots(run_dir, instance=None):
    """List all snapshots of the instance or workflow"""
    return sorted(
        run_dir.snapshot_dir(instance).iterdir(),
        key=lambda path: tuple(map(int, path.stem.split("_")[1:])),
    )


def test_source_checkpoints(tmp_path: Path, pf_active) -> None:
    """
    Test if checkpointing works as intended.

    This stays on the plain Manager/RunDir setup rather than MuscleTester:
    checkpoints apply to the whole workflow, and the tester component that
    MuscleTester wires in does not declare checkpoint support, so the
    manager rejects any config with a `checkpoints:` section as soon as the
    tester tries to connect.
    """
    data_source_path = (tmp_path / "source_component_data").absolute()
    data_sink_path = (tmp_path / "sink_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    sink_uri = f"imas:hdf5?path={data_sink_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(pf_active)
    # make config
    ymmsl_text = f"""
ymmsl_version: v0.2
models:
  test_model:
    components:
      source_component:
        description: source component
        implementation: source_component
        ports:
          o_i: [pf_active_out]
      sink_component:
        description: sink component
        implementation: sink_component
        ports:
          f_init: [pf_active_in]
    conduits:
      source_component.pf_active_out: sink_component.pf_active_in
settings:
  source_component.source_uri: {source_uri}
  source_component.iterative: true
  sink_component.sink_uri: {sink_uri}
  sink_component.sink_mode: 'w'
programs:
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
checkpoints:
  simulation_time:
  - every: 0.5
"""

    config = ymmsl.load(ymmsl_text)
    run_dir = RunDir(tmp_path / "run")
    run_dir2 = RunDir(tmp_path / "run2")
    assert all(pf_active.time == [0, 1, 2])
    for i in range(2):
        if i == 0:
            manager = Manager(config, run_dir)
            expected_time = [0, 1, 2]
        if i == 1:
            manager = Manager(config, run_dir2)
            snapshots_ymmsl = ls_snapshots(run_dir)
            assert len(snapshots_ymmsl) == 3
            config.update(ymmsl.load(snapshots_ymmsl[-1]))
            expected_time = [2]
        manager.start_instances()
        success = manager.wait()
        assert success
        assert data_sink_path.exists()
        with DBEntry(sink_uri, "r") as entry:
            assert all(entry.get("pf_active").time == expected_time)


def test_increment_existing_sink(
    muscle3_tester: MuscleTester, tmp_path: Path, core_profiles
) -> None:
    sink_path = (tmp_path / "sink_data").absolute()
    sink_uri = f"imas:hdf5?path={sink_path}"
    # pre-create the sink path so the sink component has to avoid a collision
    with DBEntry(sink_uri, "w"):
        pass

    config = f"""
ymmsl_version: v0.2
programs:
  sink_component:
    ports:
      f_init: [core_profiles_in]
    executable: python
    args: -u -m imas_muscle3.actors.sink_component
settings:
  sink_component.sink_uri: {sink_uri}
  sink_component.sink_mode: x
"""
    tester = muscle3_tester.start_implementation(config, "sink_component")

    for msg in slice_messages(core_profiles, "core_profiles"):
        tester.send("core_profiles_in", msg)

    # Shut the implementation down so its sink DBEntry is flushed and closed
    # before we read it back.
    muscle3_tester.cleanup()

    new_sink_path = sink_path.with_name(
        f"{sink_path.stem}_1{sink_path.suffix}"
    )
    assert new_sink_path.exists()
    new_sink_uri = f"imas:hdf5?path={new_sink_path}"
    with DBEntry(new_sink_uri, "r") as entry:
        assert all(entry.get("core_profiles").time == core_profiles.time)

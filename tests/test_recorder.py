import pytest
import xarray as xr
import ymmsl
from imas import DBEntry
from libmuscle import Message
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.recorder.base import Recorder
from imas_muscle3.utils import ids_name_from_port

# --- port -> IDS name -----------------------------------------------------


def test_ids_name_from_port_strips_suffix():
    assert ids_name_from_port("equilibrium_in") == "equilibrium"
    assert ids_name_from_port("equilibrium") == "equilibrium"


def test_ids_name_from_port_rejects_unknown():
    with pytest.raises(ValueError):
        ids_name_from_port("not_an_ids_in")


# --- occurrence splitting (format-agnostic) --------------------------------


class _FakeRecorder(Recorder):
    """Records which occurrence bases were opened; writes nothing to disk."""

    def __init__(self, store_dir, ids_name, extract, profile, opened):
        super().__init__(store_dir, ids_name, extract, profile)
        self._opened = opened

    def _open_occurrence(self, base):
        self._opened.append(base.name)

    def _write(self, datasets):
        return ""

    def _close_occurrence(self):
        pass


def _recorder(tmp_path, opened):
    return _FakeRecorder(
        tmp_path, "equilibrium", lambda ids: {}, "cfg.py", opened
    )


def test_recorder_splits_occurrences_on_restart(tmp_path, equilibrium):
    """A backward time step / end-of-stream starts a new occurrence."""
    opened = []
    rec = _recorder(tmp_path, opened)
    data = equilibrium.serialize()
    # iteration 0: t=0,1 (1 ends the stream); iteration 1: t=0,1 (time resets).
    rec.handle(Message(0.0, 1.0, data=data))
    rec.handle(Message(1.0, None, data=data))
    rec.handle(Message(0.0, 1.0, data=data))
    rec.handle(Message(1.0, None, data=data))
    rec.close()
    assert opened == ["0000", "0001"]


def test_recorder_one_occurrence_for_monotonic(tmp_path, equilibrium):
    """A single monotonic trace stays one occurrence (no spurious split)."""
    opened = []
    rec = _recorder(tmp_path, opened)
    data = equilibrium.serialize()
    rec.handle(Message(0.0, 1.0, data=data))
    rec.handle(Message(1.0, 2.0, data=data))
    rec.handle(Message(2.0, None, data=data))
    rec.close()
    assert opened == ["0000"]


# --- checkpoint/resume bookkeeping ------------------------------------------


def test_recorder_state_roundtrip_resumes_same_occurrence(
    tmp_path, equilibrium
):
    """A fresh Recorder restored from a mid-stream get_state() continues the
    same occurrence, rather than starting over at 0000."""
    opened = []
    rec = _recorder(tmp_path, opened)
    data = equilibrium.serialize()
    rec.handle(Message(0.0, 1.0, data=data))
    rec.handle(Message(1.0, None, data=data))  # ends occurrence 0000
    rec.handle(Message(0.0, 1.0, data=data))  # starts occurrence 0001

    state = rec.get_state()
    assert state == {
        "occurrence": 1,
        "last_time": 0.0,
        "prev_ended": False,
        "is_open": True,
    }

    # A new process would build a fresh Recorder and restore its state.
    resumed_opened = []
    resumed = _recorder(tmp_path, resumed_opened)
    resumed.restore_state(state)
    assert resumed_opened == ["0001"]  # reopened the still-open occurrence

    resumed.handle(Message(1.0, None, data=data))  # continues, doesn't split
    resumed.close()
    assert resumed_opened == ["0001"]


# --- integration: two timelines -> one recorder ----------------------------

# The recorder's config file: one single-time dataset per received IDS.
_CONFIG = """
import xarray as xr


def extract(ids):
    t = float(ids.time[0])
    return {
        ids.metadata.name: xr.Dataset(
            {"t": ("time", [t])}, coords={"time": [t]}
        )
    }
"""


def _ymmsl(eq_uri, cp_uri, store_path, config_path):
    return f"""
ymmsl_version: v0.2
models:
  test_recorder:
    components:
      eq_source:
        description: equilibrium source component
        implementation: source_component
        ports:
          o_i: [equilibrium_out]
      cp_source:
        description: core_profiles source component
        implementation: source_component
        ports:
          o_i: [core_profiles_out]
      rec:
        description: recorder component
        implementation: recorder_component
        ports:
          s: [equilibrium_in, core_profiles_in]
    conduits:
      eq_source.equilibrium_out: rec.equilibrium_in
      cp_source.core_profiles_out: rec.core_profiles_in
settings:
  eq_source.source_uri: {eq_uri}
  cp_source.source_uri: {cp_uri}
  rec.store_path: {store_path}
  rec.config: {config_path}
programs:
  recorder_component:
    executable: python
    args: -u -m imas_muscle3.actors.recorder_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  test_recorder.eq_source:
    threads: 1
  test_recorder.cp_source:
    threads: 1
  test_recorder.rec:
    threads: 1
"""


def test_records_two_timelines(tmp_path, equilibrium, core_profiles):
    eq_uri = f"imas:hdf5?path={(tmp_path / 'eq_data').absolute()}"
    cp_uri = f"imas:hdf5?path={(tmp_path / 'cp_data').absolute()}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(equilibrium)
    with DBEntry(cp_uri, "w") as entry:
        entry.put(core_profiles)
    config_path = tmp_path / "extract_time.py"
    config_path.write_text(_CONFIG)

    store_path = (tmp_path / "store").absolute()
    config = ymmsl.load(_ymmsl(eq_uri, cp_uri, store_path, config_path))
    manager = Manager(config, RunDir(tmp_path / "run"))
    manager.start_instances()
    assert manager.wait()

    # Each timeline streams into one occurrence -> one store per port.
    for port, ids_name in (
        ("equilibrium_in", "equilibrium"),
        ("core_profiles_in", "core_profiles"),
    ):
        occurrences = sorted((store_path / port).glob("*.zarr"))
        assert [o.name for o in occurrences] == ["0000.zarr"]
        ds = xr.open_zarr(occurrences[0], group=ids_name, consolidated=False)
        assert list(ds.time.values) == list(equilibrium.time)


# --- integration: checkpoint + resume ---------------------------------------


def ls_snapshots(run_dir, instance=None):
    """List all snapshots of the instance or workflow."""
    return sorted(
        run_dir.snapshot_dir(instance).iterdir(),
        key=lambda path: tuple(map(int, path.stem.split("_")[1:])),
    )


def _checkpoint_ymmsl(eq_uri, store_path, config_path):
    return f"""
ymmsl_version: v0.2
models:
  test_recorder:
    components:
      eq_source:
        description: equilibrium source component
        implementation: source_component
        ports:
          o_i: [equilibrium_out]
      rec:
        description: recorder component
        implementation: recorder_component
        ports:
          s: [equilibrium_in]
    conduits:
      eq_source.equilibrium_out: rec.equilibrium_in
settings:
  eq_source.source_uri: {eq_uri}
  eq_source.iterative: true
  rec.store_path: {store_path}
  rec.config: {config_path}
programs:
  recorder_component:
    executable: python
    args: -u -m imas_muscle3.actors.recorder_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  test_recorder.eq_source:
    threads: 1
  test_recorder.rec:
    threads: 1
checkpoints:
  simulation_time:
  - every: 0.5
"""


def test_recorder_resumes_from_checkpoint(tmp_path, equilibrium):
    """Kill the workflow after a checkpoint and resume it in a fresh run
    directory: the recorder must continue the same occurrence rather than
    restarting (and overwriting) it from 0000."""
    eq_uri = f"imas:hdf5?path={(tmp_path / 'eq_data').absolute()}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(equilibrium)
    config_path = tmp_path / "extract_time.py"
    config_path.write_text(_CONFIG)
    store_path = (tmp_path / "store").absolute()

    config = ymmsl.load(
        _checkpoint_ymmsl(eq_uri, store_path, config_path)
    )
    run_dir = RunDir(tmp_path / "run")
    run_dir2 = RunDir(tmp_path / "run2")

    manager = Manager(config, run_dir)
    manager.start_instances()
    assert manager.wait()

    snapshots = ls_snapshots(run_dir)
    assert snapshots  # at least one global checkpoint was taken

    # Simulate a crash-and-restart: a fresh manager, in a fresh run
    # directory, resuming from the last global checkpoint.
    config.update(ymmsl.load(snapshots[-1]))
    manager2 = Manager(config, run_dir2)
    manager2.start_instances()
    assert manager2.wait()

    # One continuous occurrence holding all three time points -- not
    # restarted (which would truncate/overwrite it) or duplicated.
    occurrences = sorted((store_path / "equilibrium_in").glob("*.zarr"))
    assert [o.name for o in occurrences] == ["0000.zarr"]
    ds = xr.open_zarr(occurrences[0], group="equilibrium", consolidated=False)
    assert list(ds.time.values) == list(equilibrium.time)

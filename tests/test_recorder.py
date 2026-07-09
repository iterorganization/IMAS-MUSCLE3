import pytest
import xarray as xr
from imas import DBEntry
from libmuscle import Message
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.actors._tap_base import (
    OccurrenceRecorder,
    ids_name_from_port,
)
from tests.ymmsl_helpers import load_config

# --- port -> IDS name -----------------------------------------------------


def test_ids_name_from_port_strips_suffix():
    assert ids_name_from_port("equilibrium_in") == "equilibrium"
    assert ids_name_from_port("equilibrium") == "equilibrium"


def test_ids_name_from_port_rejects_unknown():
    with pytest.raises(ValueError):
        ids_name_from_port("not_an_ids_in")


# --- occurrence splitting (sink-agnostic) ----------------------------------


class _NullSink:
    def write(self, msg):
        return ""

    def close(self):
        pass


def _recorder(tmp_path, created):
    def factory(base, ids_name):
        created.append(base.name)
        return _NullSink()

    return OccurrenceRecorder(tmp_path, "equilibrium", factory)


def test_recorder_splits_occurrences_on_restart(tmp_path):
    """A backward time step / end-of-stream starts a new occurrence."""
    created = []
    rec = _recorder(tmp_path, created)
    # iteration 0: t=0,1 (1 ends the stream); iteration 1: t=0,1 (time resets).
    rec.handle(Message(0.0, 1.0, data=b""))
    rec.handle(Message(1.0, None, data=b""))
    rec.handle(Message(0.0, 1.0, data=b""))
    rec.handle(Message(1.0, None, data=b""))
    rec.close()
    assert created == ["0000", "0001"]


def test_recorder_one_occurrence_for_monotonic(tmp_path):
    """A single monotonic trace stays one occurrence (no spurious split)."""
    created = []
    rec = _recorder(tmp_path, created)
    rec.handle(Message(0.0, 1.0, data=b""))
    rec.handle(Message(1.0, 2.0, data=b""))
    rec.handle(Message(2.0, None, data=b""))
    rec.close()
    assert created == ["0000"]


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
ymmsl_version: v0.1
model:
  name: test_recorder
  components:
    eq_source:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    cp_source:
      implementation: source_component
      ports:
        o_i: [core_profiles_out]
    rec:
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
implementations:
  recorder_component:
    executable: python
    args: -u -m imas_muscle3.actors.recorder_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  eq_source: {{threads: 1}}
  cp_source: {{threads: 1}}
  rec: {{threads: 1}}
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
    config = load_config(_ymmsl(eq_uri, cp_uri, store_path, config_path))
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

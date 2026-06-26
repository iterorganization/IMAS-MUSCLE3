import msgpack
import pytest
import xarray as xr
from imas import DBEntry, IDSFactory
from libmuscle import Message
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.actors._tap_base import (
    OccurrenceRecorder,
    ids_name_from_port,
)
from imas_muscle3.actors.recorder_component import DBEntrySink, RawSink
from tests.ymmsl_helpers import load_config


# --- port -> IDS name -----------------------------------------------------


def test_ids_name_from_port_strips_suffix():
    assert ids_name_from_port("equilibrium_in") == "equilibrium"
    assert ids_name_from_port("equilibrium") == "equilibrium"


def test_ids_name_from_port_rejects_unknown():
    with pytest.raises(ValueError):
        ids_name_from_port("not_an_ids_in")


# --- sinks ----------------------------------------------------------------


def test_dbentry_sink_roundtrip(tmp_path, equilibrium):
    base = tmp_path / "equilibrium_in" / "0000"
    base.parent.mkdir(parents=True)
    sink = DBEntrySink(base, "equilibrium")
    sink.write(Message(0.0, None, data=equilibrium.serialize()))
    sink.close()

    with DBEntry(f"imas:hdf5?path={base}", "r") as entry:
        assert all(entry.get("equilibrium").time == equilibrium.time)


def test_dbentry_sink_per_message(tmp_path, equilibrium):
    base = tmp_path / "equilibrium_in" / "0000"
    base.parent.mkdir(parents=True)
    sink = DBEntrySink(base, "equilibrium", per_message=True)
    sink.write(Message(0.0, None, data=equilibrium.serialize()))
    sink.write(Message(1.0, None, data=equilibrium.serialize()))
    sink.close()

    # Each message is its own complete, immediately-readable DBEntry.
    e0 = base.parent / "0000_00000000"
    e1 = base.parent / "0000_00000001"
    assert e0.is_dir() and e1.is_dir()
    with DBEntry(f"imas:hdf5?path={e0}", "r") as entry:
        assert all(entry.get("equilibrium").time == equilibrium.time)


def test_raw_sink_frames_and_replays(tmp_path, equilibrium):
    base = tmp_path / "equilibrium_in" / "0000"
    base.parent.mkdir(parents=True)
    data = bytes(equilibrium.serialize())
    sink = RawSink(base, "equilibrium")
    sink.write(Message(0.0, 1.0, data=data))
    sink.write(Message(1.0, None, data=data))
    sink.close()

    with open(base.parent / "0000.msgpack", "rb") as fh:
        records = list(msgpack.Unpacker(fh, raw=False))
    # The MUSCLE3 frame is preserved alongside the undecoded payload.
    assert [r["t"] for r in records] == [0.0, 1.0]
    assert [r["next_t"] for r in records] == [1.0, None]
    assert records[0]["data"] == data

    eq = IDSFactory().new("equilibrium")
    eq.deserialize(records[0]["data"])
    assert all(eq.time == equilibrium.time)


# --- occurrence splitting (format-agnostic; uses the cheap RawSink) -------


def _raw_recorder(store_dir):
    return OccurrenceRecorder(store_dir, "equilibrium", RawSink)


def _msg(t, nxt, data):
    return Message(t, nxt, data=data)


def test_recorder_splits_occurrences_on_restart(tmp_path, equilibrium):
    """A backward time step / end-of-stream starts a new occurrence."""
    data = equilibrium.serialize()
    store_dir = tmp_path / "equilibrium_in"
    rec = _raw_recorder(store_dir)
    # iteration 0: t=0,1 (1 ends the stream); iteration 1: t=0,1 (time resets).
    rec.handle(0, _msg(0.0, 1.0, data))
    rec.handle(1, _msg(1.0, None, data))
    rec.handle(2, _msg(0.0, 1.0, data))
    rec.handle(3, _msg(1.0, None, data))
    rec.close()

    assert (store_dir / "0000.msgpack").is_file()
    assert (store_dir / "0001.msgpack").is_file()
    assert not (store_dir / "0002.msgpack").exists()


def test_recorder_one_occurrence_for_monotonic(tmp_path, equilibrium):
    """A single monotonic trace stays one occurrence (no spurious split)."""
    data = equilibrium.serialize()
    store_dir = tmp_path / "equilibrium_in"
    rec = _raw_recorder(store_dir)
    rec.handle(0, _msg(0.0, 1.0, data))
    rec.handle(1, _msg(1.0, 2.0, data))
    rec.handle(2, _msg(2.0, None, data))
    rec.close()

    assert (store_dir / "0000.msgpack").is_file()
    assert not (store_dir / "0001.msgpack").exists()


# --- integration: two timelines -> one recorder, per format ---------------


def _ymmsl(fmt, eq_uri, cp_uri, store_path):
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
  rec.format: {fmt}
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


@pytest.mark.parametrize("fmt", ["imas", "raw", "distill"])
def test_records_two_timelines(fmt, tmp_path, equilibrium, core_profiles):
    eq_uri = f"imas:hdf5?path={(tmp_path / 'eq_data').absolute()}"
    cp_uri = f"imas:hdf5?path={(tmp_path / 'cp_data').absolute()}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(equilibrium)
    with DBEntry(cp_uri, "w") as entry:
        entry.put(core_profiles)

    store_path = (tmp_path / "store").absolute()
    config = load_config(_ymmsl(fmt, eq_uri, cp_uri, store_path))
    manager = Manager(config, RunDir(tmp_path / "run"))
    manager.start_instances()
    assert manager.wait()

    # Each timeline streams into one occurrence -> one store per port.
    eq_occ = sorted((store_path / "equilibrium_in").glob("0000*"))
    cp_occ = sorted((store_path / "core_profiles_in").glob("0000*"))
    assert len(eq_occ) == 1 and len(cp_occ) == 1

    if fmt == "imas":
        with DBEntry(f"imas:hdf5?path={eq_occ[0]}", "r") as entry:
            assert all(entry.get("equilibrium").time == equilibrium.time)
    elif fmt == "raw":
        with open(eq_occ[0], "rb") as fh:
            records = list(msgpack.Unpacker(fh, raw=False))
        assert [r["t"] for r in records] == list(equilibrium.time)
    else:  # distill
        eq = xr.open_zarr(eq_occ[0], group="equilibrium", consolidated=False)
        ip = eq["time_slice.global_quantities.ip"]
        assert list(eq.time.values) == list(equilibrium.time)
        assert float(ip.values[0]) == pytest.approx(1e6)

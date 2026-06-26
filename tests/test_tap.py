import multiprocessing

import pytest
from imas import DBEntry
from libmuscle import Message
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.actors.tap_component import DBEntrySink, ids_name_from_port
from tests.ymmsl_helpers import load_config

"""Force 'spawn' start method to avoid deadlocks with pytest."""
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


# --- unit tests -----------------------------------------------------------


def test_ids_name_from_port_strips_suffix():
    assert ids_name_from_port("equilibrium_in") == "equilibrium"
    assert ids_name_from_port("equilibrium") == "equilibrium"


def test_ids_name_from_port_rejects_unknown():
    with pytest.raises(ValueError):
        ids_name_from_port("not_an_ids_in")


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
    import msgpack
    from imas import IDSFactory

    from imas_muscle3.actors.raw_component import RawSink

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


# --- integration test: two timelines -> one tap ---------------------------


def _ymmsl_two_timelines(eq_uri, pf_uri, store_path):
    return f"""
ymmsl_version: v0.1
model:
  name: test_tap
  components:
    eq_source:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    pf_source:
      implementation: source_component
      ports:
        o_i: [pf_active_out]
    tap:
      implementation: tap_component
      ports:
        s: [equilibrium_in, pf_active_in]
  conduits:
    eq_source.equilibrium_out: tap.equilibrium_in
    pf_source.pf_active_out: tap.pf_active_in
settings:
  eq_source.source_uri: {eq_uri}
  pf_source.source_uri: {pf_uri}
  tap.store_path: {store_path}
implementations:
  tap_component:
    executable: python
    args: -u -m imas_muscle3.actors.tap_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  eq_source:
    threads: 1
  cp_source:
    threads: 1
  tap:
    threads: 1
"""


def test_tap_records_two_timelines(tmp_path, equilibrium, pf_active):
    eq_path = (tmp_path / "eq_source_data").absolute()
    pf_path = (tmp_path / "pf_source_data").absolute()
    eq_uri = f"imas:hdf5?path={eq_path}"
    pf_uri = f"imas:hdf5?path={pf_path}"
    with DBEntry(eq_uri, "w") as entry:
        entry.put(equilibrium)
    with DBEntry(pf_uri, "w") as entry:
        entry.put(pf_active)

    store_path = (tmp_path / "tap_store").absolute()
    config = load_config(_ymmsl_two_timelines(eq_uri, pf_uri, store_path))
    run_dir = RunDir(tmp_path / "run")
    manager = Manager(config, run_dir)
    manager.start_instances()
    assert manager.wait()

    # Each timeline streams its slices into one occurrence -> one DBEntry per
    # port, holding that occurrence's full trace.
    eq_occ = sorted((store_path / "equilibrium_in").glob("*"))
    pf_occ = sorted((store_path / "pf_active_in").glob("*"))
    assert len(eq_occ) == 1
    assert len(pf_occ) == 1

    with DBEntry(f"imas:hdf5?path={eq_occ[0]}", "r") as entry:
        assert all(entry.get("equilibrium").time == equilibrium.time)
    with DBEntry(f"imas:hdf5?path={pf_occ[0]}", "r") as entry:
        assert all(entry.get("pf_active").time == pf_active.time)

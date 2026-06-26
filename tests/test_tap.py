import multiprocessing

import pytest
from imas import DBEntry
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.actors.tap_component import (
    ids_name_from_port,
    record_message,
)
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


def test_record_message_roundtrip(tmp_path, equilibrium):
    store_path = tmp_path / "store"
    data = equilibrium.serialize()
    record_message(store_path, "equilibrium_in", "equilibrium", data, seq=0)

    msg_dir = store_path / "equilibrium_in" / "00000000"
    assert msg_dir.is_dir()
    uri = f"imas:hdf5?path={msg_dir}"
    with DBEntry(uri, "r") as entry:
        assert all(entry.get("equilibrium").time == equilibrium.time)


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

    # Each timeline is iterated over its 3 time points -> 3 per-message
    # DBEntries per port.
    eq_msgs = sorted((store_path / "equilibrium_in").glob("*"))
    pf_msgs = sorted((store_path / "pf_active_in").glob("*"))
    assert len(eq_msgs) == len(equilibrium.time)
    assert len(pf_msgs) == len(pf_active.time)

    # The recorded slices read back as the right IDS.
    with DBEntry(f"imas:hdf5?path={eq_msgs[0]}", "r") as entry:
        assert entry.get("equilibrium").time.size >= 1
    with DBEntry(f"imas:hdf5?path={pf_msgs[0]}", "r") as entry:
        assert entry.get("pf_active").time.size >= 1

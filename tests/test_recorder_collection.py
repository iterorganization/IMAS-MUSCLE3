import pytest
import xarray as xr
from libmuscle import Message

from imas_muscle3.recorder.base import Recorder
from imas_muscle3.recorder.collection import (
    LiveState,
    RecorderCollection,
    load_extract_config,
    snapshot_config,
)

# --- config loading ---------------------------------------------------------


def test_load_extract_fn(tmp_path, equilibrium):
    config = tmp_path / "config.py"
    config.write_text(
        "import xarray as xr\n"
        "def extract(ids):\n"
        "    return {'marker': xr.Dataset(\n"
        "        {'ip': ('time', [float(ids.time_slice[0]"
        ".global_quantities.ip)])},\n"
        "        coords={'time': [float(ids.time[0])]})}\n"
    )
    extract = load_extract_config(str(config))
    out = extract(equilibrium)
    assert set(out) == {"marker"}
    assert float(out["marker"]["ip"].values[0]) == pytest.approx(1e6)


def test_load_extract_state_class(tmp_path, equilibrium):
    config = tmp_path / "plot_file.py"
    config.write_text(
        "import xarray as xr\n"
        "from imas_muscle3.visualization.base_state import BaseState\n"
        "class State(BaseState):\n"
        "    def extract(self, ids):\n"
        "        self.data[ids.metadata.name] = xr.Dataset(\n"
        "            {'n': ('time', [float(len(ids.time_slice))])},\n"
        "            coords={'time': [float(ids.time[0])]})\n"
    )
    extract = load_extract_config(str(config))
    out = extract(equilibrium)
    assert set(out) == {"equilibrium"}
    assert float(out["equilibrium"]["n"].values[0]) == 3.0


def test_load_extract_config_rejects_other_files(tmp_path):
    config = tmp_path / "not_a_config.py"
    config.write_text("x = 1\n")
    with pytest.raises(NameError):
        load_extract_config(str(config))


def test_snapshot_config_copies_next_to_data(tmp_path):
    config = tmp_path / "cfg.py"
    config.write_text("x = 1\n")
    store_path = tmp_path / "store"
    store_path.mkdir()

    snapshot = snapshot_config(config, store_path)
    assert snapshot == store_path / "cfg.py"
    # Editing the original leaves the snapshot untouched...
    config.write_text("x = 2\n")
    assert snapshot.read_text() == "x = 1\n"
    # ...until the next run re-snapshots it.
    assert snapshot_config(config, store_path).read_text() == "x = 2\n"
    # Config already next to the data: returned as-is, no self-copy.
    assert snapshot_config(snapshot, store_path) == snapshot


# --- live state --------------------------------------------------------------


def _ds(t, value):
    return xr.Dataset({"v": ("time", [value])}, coords={"time": [t]})


def test_live_state_accumulates_along_time():
    state = LiveState()
    state.update({"equilibrium": _ds(0.0, 1.0)})
    state.update({"equilibrium": _ds(1.0, 2.0)})
    assert list(state.data["equilibrium"].time.values) == [0.0, 1.0]
    assert list(state.data["equilibrium"]["v"].values) == [1.0, 2.0]


# --- collection wiring -------------------------------------------------------

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


class _RecordingRecorder(Recorder):
    """A no-op Recorder that just remembers what it was asked to write."""

    def __init__(self, store_dir, ids_name, extract, profile, log):
        super().__init__(store_dir, ids_name, extract, profile)
        self._log = log

    def _open_occurrence(self, base):
        pass

    def _write(self, datasets):
        self._log.append(datasets)
        return ""

    def _close_occurrence(self):
        pass


def test_collection_routes_per_port_and_updates_live_state(
    tmp_path, equilibrium
):
    config = tmp_path / "config.py"
    config.write_text(_CONFIG)
    store_path = tmp_path / "store"
    store_path.mkdir()
    log = []

    def make_recorder(store_dir, ids_name, extract, profile):
        return _RecordingRecorder(store_dir, ids_name, extract, profile, log)

    collection = RecorderCollection(
        store_path, config, {"equilibrium_in": "equilibrium"}, make_recorder
    )
    assert collection.config_snapshot == store_path / "config.py"
    assert set(collection.live_state) == {"equilibrium_in"}
    assert collection.live_state["equilibrium_in"].data == {}

    collection.handle(
        "equilibrium_in", Message(0.0, None, data=equilibrium.serialize())
    )

    assert len(log) == 1
    assert set(collection.live_state["equilibrium_in"].data) == {"equilibrium"}

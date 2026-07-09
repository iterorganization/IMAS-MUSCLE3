import numpy as np
import pytest
import xarray as xr
from libmuscle import Message

from imas_muscle3.distill import ZarrSink, group_name
from imas_muscle3.distill.sink import DistillSink, load_extract_config
from imas_muscle3.distill.zarr_sink import read_root_attrs

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
    # A visualization plot file's State class works as-is: each message goes
    # through a fresh instance and its accumulated datasets are recorded.
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


# --- distill sink -----------------------------------------------------------


def test_distill_sink_writes_and_stamps(tmp_path, equilibrium):
    def extract(ids):
        t = float(ids.time[0])
        return {
            "equilibrium": xr.Dataset(
                {"t": ("time", [t])}, coords={"time": [t]}
            )
        }

    sink = DistillSink(
        tmp_path / "0002", "equilibrium", extract, profile="cfg.py"
    )
    sink.write(Message(0.0, None, data=equilibrium.serialize()))
    sink.close()

    store = tmp_path / "0002.zarr"
    ds = xr.open_zarr(store, group="equilibrium", consolidated=False)
    assert list(ds.time.values) == [0.0]
    # Root attrs let a viewer group stores and find the matching plots.
    attrs = read_root_attrs(store)
    assert attrs["occurrence"] == 2
    assert attrs["distill_profile"].endswith("cfg.py")


# --- zarr sink ------------------------------------------------------------


def test_group_name_is_slash_free():
    assert group_name("equilibrium/time_slice[0]/global_quantities/ip") == (
        "equilibrium.time_slice[0].global_quantities.ip"
    )


def _single_1d(t, values):
    arr = np.asarray(values, dtype=float)
    return xr.Dataset(
        {
            "value": (("time", "dim0"), arr[np.newaxis, :]),
            "coord0": (
                ("time", "dim0"),
                np.arange(len(arr), dtype=float)[np.newaxis, :],
            ),
        },
        coords={"time": [t]},
        attrs={"full_path": "x/y"},
    )


def test_zarr_sink_combines_along_time(tmp_path):
    store = tmp_path / "core_profiles_in.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))
    sink.append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert list(ds.time.values) == [0.0, 1.0]
    assert ds["value"].shape == (2, 8)
    assert ds.attrs["full_path"] == "x/y"


def test_zarr_sink_writes_whole_trace(tmp_path):
    # A single message carrying a whole trace (time>1) is written in one go.
    store = tmp_path / "equilibrium_in.zarr"
    ds = xr.Dataset(
        {"value": (("time", "dim0"), np.ones((49, 8)))},
        coords={"time": np.arange(49.0)},
        attrs={"full_path": "x/y"},
    )
    sink = ZarrSink(store)
    sink.append("x/y", ds)
    sink.close()
    out = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert out["value"].shape == (49, 8)


def test_zarr_sink_pads_ragged_profiles(tmp_path):
    store = tmp_path / "store.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))
    # A shorter later slice is NaN-padded up to the max width.
    sink.append("x/y", _single_1d(1.0, np.full(5, 3.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert ds["value"].shape == (2, 8)
    second = ds["value"].values[1]
    assert list(second[:5]) == [3.0] * 5
    assert np.isnan(second[5:]).all()


def test_zarr_sink_combines_gaps(tmp_path):
    # Some messages carry a var, others don't (e.g. an empty profiles_1d
    # early in a run). The store must still open: union time axis, NaN where
    # the var is absent.
    store = tmp_path / "core_profiles_in.zarr"
    sink = ZarrSink(store)
    sink.append("x/y", _single_1d(0.0, np.ones(8)))  # has 'value'
    sink.append(
        "x/y",
        xr.Dataset({"other": ("time", [9.0])}, coords={"time": [0.5]}),
    )  # gap: no 'value'
    sink.append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    sink.close()

    ds = xr.open_zarr(store, group=group_name("x/y"), consolidated=False)
    assert list(ds.time.values) == [0.0, 0.5, 1.0]
    assert ds["value"].shape == (3, 8)
    assert np.isnan(ds["value"].values[1]).all()  # gap NaN-filled
    assert np.isnan(ds["other"].values[0]) and ds["other"].values[1] == 9.0

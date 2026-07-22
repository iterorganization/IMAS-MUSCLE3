import numpy as np
import xarray as xr
from libmuscle import Message

from imas_muscle3.recorder.zarr_recorder import (
    ZarrRecorder,
    group_name,
    read_root_attrs,
)


def _open(tmp_path, name):
    rec = ZarrRecorder(tmp_path, "equilibrium", lambda ids: {}, "cfg.py")
    rec._open_occurrence(tmp_path / name)
    return rec


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


def test_zarr_recorder_combines_along_time(tmp_path):
    rec = _open(tmp_path, "0000")
    rec._append("x/y", _single_1d(0.0, np.ones(8)))
    rec._append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    rec._close_occurrence()

    ds = xr.open_zarr(
        tmp_path / "0000.zarr",
        group=group_name("x/y"),
        consolidated=False,
    )
    assert list(ds.time.values) == [0.0, 1.0]
    assert ds["value"].shape == (2, 8)
    assert ds.attrs["full_path"] == "x/y"


def test_zarr_recorder_writes_whole_trace(tmp_path):
    # A single message carrying a whole trace (time>1) is written in one go.
    rec = _open(tmp_path, "0000")
    ds = xr.Dataset(
        {"value": (("time", "dim0"), np.ones((49, 8)))},
        coords={"time": np.arange(49.0)},
        attrs={"full_path": "x/y"},
    )
    rec._append("x/y", ds)
    rec._close_occurrence()
    out = xr.open_zarr(
        tmp_path / "0000.zarr",
        group=group_name("x/y"),
        consolidated=False,
    )
    assert out["value"].shape == (49, 8)


def test_zarr_recorder_pads_ragged_profiles(tmp_path):
    rec = _open(tmp_path, "0000")
    rec._append("x/y", _single_1d(0.0, np.ones(8)))
    # A shorter later slice is NaN-padded up to the max width.
    rec._append("x/y", _single_1d(1.0, np.full(5, 3.0)))
    rec._close_occurrence()

    ds = xr.open_zarr(
        tmp_path / "0000.zarr", group=group_name("x/y"), consolidated=False
    )
    assert ds["value"].shape == (2, 8)
    second = ds["value"].values[1]
    assert list(second[:5]) == [3.0] * 5
    assert np.isnan(second[5:]).all()


def test_zarr_recorder_combines_gaps(tmp_path):
    # A var absent from some messages: union time axis, NaN where missing.
    rec = _open(tmp_path, "0000")
    rec._append("x/y", _single_1d(0.0, np.ones(8)))  # has 'value'
    rec._append(
        "x/y",
        xr.Dataset({"other": ("time", [9.0])}, coords={"time": [0.5]}),
    )  # gap: no 'value'
    rec._append("x/y", _single_1d(1.0, np.full(8, 2.0)))
    rec._close_occurrence()

    ds = xr.open_zarr(
        tmp_path / "0000.zarr",
        group=group_name("x/y"),
        consolidated=False,
    )
    assert list(ds.time.values) == [0.0, 0.5, 1.0]
    assert ds["value"].shape == (3, 8)
    assert np.isnan(ds["value"].values[1]).all()  # gap NaN-filled
    assert np.isnan(ds["other"].values[0]) and ds["other"].values[1] == 9.0


# --- checkpoint/resume: rehydrating the rebuild buffer from disk -----------


def test_reopen_occurrence_rehydrates_buffer_for_later_rebuild(tmp_path):
    """A schema mismatch after resume must still rebuild from the FULL
    history, including messages written before the checkpoint -- not just
    what arrived since. Regression test for silently losing pre-checkpoint
    data on a post-resume rebuild."""
    # Before the (simulated) checkpoint: one message on disk, no _close.
    rec1 = _open(tmp_path, "0000")
    rec1._append("x/y", _single_1d(0.0, np.ones(8)))

    # Resume: a brand-new ZarrRecorder, as a fresh process would build.
    rec2 = ZarrRecorder(tmp_path, "equilibrium", lambda ids: {}, "cfg.py")
    rec2._reopen_occurrence(tmp_path / "0000")
    assert list(rec2._buffers["x.y"][0]["value"].values[0]) == [1.0] * 8

    # Post-resume message has a ragged (narrower) schema: triggers _combine.
    rec2._append("x/y", _single_1d(1.0, np.full(5, 3.0)))
    rec2._close_occurrence()

    ds = xr.open_zarr(
        tmp_path / "0000.zarr", group=group_name("x/y"), consolidated=False
    )
    assert list(ds.time.values) == [0.0, 1.0]
    assert ds["value"].shape == (2, 8)
    # Pre-checkpoint row survived the rebuild...
    assert list(ds["value"].values[0]) == [1.0] * 8
    # ...alongside the post-resume, NaN-padded row.
    second = ds["value"].values[1]
    assert list(second[:5]) == [3.0] * 5
    assert np.isnan(second[5:]).all()


def test_reopen_occurrence_on_missing_store_starts_empty(tmp_path):
    """An occurrence that was opened but never written to (empty timeline)
    has no store on disk yet; resuming it must not fail."""
    rec = ZarrRecorder(tmp_path, "equilibrium", lambda ids: {}, "cfg.py")
    rec._reopen_occurrence(tmp_path / "0000")
    assert rec._buffers == {}
    assert rec._sig == {}


# --- writing via handle(), including root-attr stamping ---------------------


def test_zarr_recorder_writes_and_stamps(tmp_path, equilibrium):
    def extract(ids):
        t = float(ids.time[0])
        return {
            "equilibrium": xr.Dataset(
                {"t": ("time", [t])}, coords={"time": [t]}
            )
        }

    rec = ZarrRecorder(tmp_path, "equilibrium", extract, profile="cfg.py")
    rec.handle(Message(0.0, None, data=equilibrium.serialize()))
    rec.close()

    store = tmp_path / "0000.zarr"
    ds = xr.open_zarr(store, group="equilibrium", consolidated=False)
    assert list(ds.time.values) == [0.0]
    attrs = read_root_attrs(store)
    assert attrs["occurrence"] == 0
    assert attrs["distill_profile"] == "cfg.py"

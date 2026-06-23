"""Equilibrium overview profile — a worked example of the profile contract.

Port of the live ``pds.py`` equilibrium dashboard to the record-then-view model:
``extract`` distills the quantities once per slice, ``plot`` draws the bespoke
overview the generic browser cannot (the poloidal cross-section overlays the
separatrix and flux contours; profiles and global quantities sit alongside).

It distills, when present in the equilibrium IDS:

* ``derived/separatrix`` — the boundary outline (``r``, ``z``), a parametric
  curve the generic viewer cannot draw;
* ``derived/global`` — ``ip`` and ``beta_tor`` over time;
* ``derived/profiles_1d`` — the ``f_df_dpsi`` (ff') and ``dpressure_dpsi`` (p')
  profiles against ``psi``;
* ``derived/psi_grid`` — the unstructured (``r``, ``z``, ``psi``) grid from the
  first GGD, from which ``plot`` computes flux contours.

Every block is guarded, so it works on a minimal equilibrium (just a boundary)
as well as a full one.

Use it from a workflow with::

    settings:
      distill.config: <path>/equilibrium_overview.py
"""

import logging

import holoviews as hv
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

hv.extension("bokeh")


# --- custom distiller -------------------------------------------------------


def _curve_dataset(t, r, z):
    return xr.Dataset(
        {"r": (("time", "pt"), np.asarray(r, float)[None, :]),
         "z": (("time", "pt"), np.asarray(z, float)[None, :])},
        coords={"time": [t]},
        attrs={"kind": "curve_rz"},
    )


def extract(ids):
    """Distill derived equilibrium quantities for one time slice."""
    out = {}
    t = float(ids.time[0])
    ts = ids.time_slice[0]

    outline = ts.boundary.outline
    if len(outline.r):
        out["derived/separatrix"] = _curve_dataset(t, outline.r, outline.z)

    gq = ts.global_quantities
    globals_ = {}
    if gq.ip != 0.0 or gq.beta_tor != 0.0:
        globals_ = {"ip": ("time", [float(gq.ip)]),
                    "beta_tor": ("time", [float(gq.beta_tor)])}
    if globals_:
        out["derived/global"] = xr.Dataset(globals_, coords={"time": [t]})

    p1d = ts.profiles_1d
    if len(p1d.psi) and len(p1d.f_df_dpsi):
        out["derived/profiles_1d"] = xr.Dataset(
            {"f_df_dpsi": (("time", "psi"), np.asarray(p1d.f_df_dpsi)[None, :]),
             "dpressure_dpsi": (
                 ("time", "psi"),
                 np.asarray(p1d.dpressure_dpsi)[None, :],
             )},
            coords={"time": [t], "psi": (("time", "psi"),
                                         np.asarray(p1d.psi)[None, :])},
        )

    if len(ts.ggd) and len(ts.ggd[0].r) and len(ts.ggd[0].r[0].values):
        ggd = ts.ggd[0]
        out["derived/psi_grid"] = xr.Dataset(
            {"r": (("time", "node"), np.asarray(ggd.r[0].values)[None, :]),
             "z": (("time", "node"), np.asarray(ggd.z[0].values)[None, :]),
             "psi": (("time", "node"),
                     np.asarray(ggd.psi[0].values)[None, :])},
            coords={"time": [t]},
        )

    return out


# --- bespoke view -----------------------------------------------------------


def _contours(grid_ds, time_index, levels=20):
    """Flux contours from the unstructured psi grid (optional, needs mpl)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return hv.Overlay([])
    r = grid_ds["r"].isel(time=time_index).values
    z = grid_ds["z"].isel(time=time_index).values
    psi = grid_ds["psi"].isel(time=time_index).values
    tric = plt.tricontour(r, z, psi, levels=levels)
    segs = []
    for i, level in enumerate(tric.levels):
        for seg in tric.allsegs[i]:
            if len(seg) > 1:
                segs.append({"x": seg[:, 0], "y": seg[:, 1], "psi": level})
    plt.close("all")
    return hv.Contours(segs, vdims="psi").opts(
        cmap="viridis", colorbar=True, show_legend=False
    )


def _cross_section(data, time_index):
    sep = data.dataset("derived/separatrix")
    grid = data.dataset("derived/psi_grid")
    elements = []
    if grid is not None:
        elements.append(_contours(grid, time_index))
    if sep is not None:
        r = sep["r"].isel(time=time_index).values
        z = sep["z"].isel(time=time_index).values
        elements.append(
            hv.Curve((r, z)).opts(color="red", line_width=3)
        )
    if not elements:
        return hv.Curve(([], [])).opts(title="No equilibrium geometry")
    return hv.Overlay(elements).opts(
        title="Poloidal flux", xlabel="r [m]", ylabel="z [m]",
        aspect="equal", responsive=True, show_legend=False,
    )


def _time_series(data, time_index, var, label):
    g = data.dataset("derived/global")
    if g is None or var not in g:
        return hv.Curve(([], [])).opts(title=f"{label} (no data)")
    t = g["time"].values[: time_index + 1]
    y = g[var].isel(time=slice(0, time_index + 1)).values
    return hv.Curve((t, y), kdims=["time [s]"], vdims=[label]).opts(
        title=label, framewise=True, height=220, responsive=True
    )


def _profile(data, time_index, var, label):
    p = data.dataset("derived/profiles_1d")
    if p is None or var not in p:
        return hv.Curve(([], [])).opts(title=f"{label} (no data)")
    psi = p["psi"].isel(time=time_index).values
    y = p[var].isel(time=time_index).values
    return hv.Curve((psi, y), kdims=["psi"], vdims=[label]).opts(
        title=label, framewise=True, height=220, responsive=True
    )


def plot(data, time_index):
    """Bespoke equilibrium overview at ``time_index`` (see module docstring)."""
    import panel as pn

    return pn.Row(
        pn.pane.HoloViews(
            _cross_section(data, time_index), min_height=500, min_width=400
        ),
        pn.Column(
            pn.Row(
                _time_series(data, time_index, "ip", "Ip [A]"),
                _time_series(data, time_index, "beta_tor", "beta_tor"),
            ),
            pn.Row(
                _profile(data, time_index, "f_df_dpsi", "ff'"),
                _profile(data, time_index, "dpressure_dpsi", "p'"),
            ),
        ),
        sizing_mode="stretch_width",
    )

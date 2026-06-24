"""Equilibrium overview profile — a worked example of the profile contract.

Port of the live ``pds.py`` equilibrium dashboard to the record-then-view
model: ``extract`` distills the quantities once per slice, ``plot`` draws the
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

# Shared R-Z plasma diagram, so the recorded equilibrium renders the same
# poloidal cross-section as the waveform editor's shape editor. Optional dep:
# degrade to a plain separatrix curve when it isn't installed.
try:
    from waveform_editor.shape_editor import rz_plot
except Exception:  # pragma: no cover - waveform_editor is optional
    rz_plot = None


# --- custom distiller -------------------------------------------------------


def _curve_dataset(t, r, z):
    return xr.Dataset(
        {
            "r": (("time", "pt"), np.asarray(r, float)[None, :]),
            "z": (("time", "pt"), np.asarray(z, float)[None, :]),
        },
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
        globals_ = {
            "ip": ("time", [float(gq.ip)]),
            "beta_tor": ("time", [float(gq.beta_tor)]),
        }
    if globals_:
        out["derived/global"] = xr.Dataset(globals_, coords={"time": [t]})

    p1d = ts.profiles_1d
    if len(p1d.psi) and len(p1d.f_df_dpsi):
        out["derived/profiles_1d"] = xr.Dataset(
            {
                "f_df_dpsi": (
                    ("time", "psi"),
                    np.asarray(p1d.f_df_dpsi)[None, :],
                ),
                "dpressure_dpsi": (
                    ("time", "psi"),
                    np.asarray(p1d.dpressure_dpsi)[None, :],
                ),
            },
            coords={
                "time": [t],
                "psi": (("time", "psi"), np.asarray(p1d.psi)[None, :]),
            },
        )

    if len(ts.ggd) and len(ts.ggd[0].r) and len(ts.ggd[0].r[0].values):
        ggd = ts.ggd[0]
        out["derived/psi_grid"] = xr.Dataset(
            {
                "r": (("time", "node"), np.asarray(ggd.r[0].values)[None, :]),
                "z": (("time", "node"), np.asarray(ggd.z[0].values)[None, :]),
                "psi": (
                    ("time", "node"),
                    np.asarray(ggd.psi[0].values)[None, :],
                ),
            },
            coords={"time": [t]},
        )

    # Critical points (X-points / O-points) for the cross-section overlay; the
    # solver may leave contour_tree empty (e.g. inverse mode) — then skip them.
    try:
        nodes = ts.contour_tree.node
    except Exception:
        nodes = []
    if len(nodes):
        out["derived/xo_points"] = xr.Dataset(
            {
                "r": (
                    ("time", "pt"),
                    np.asarray([n.r for n in nodes], float)[None, :],
                ),
                "z": (
                    ("time", "pt"),
                    np.asarray([n.z for n in nodes], float)[None, :],
                ),
                "critical_type": (
                    ("time", "pt"),
                    np.asarray([n.critical_type for n in nodes], float)[
                        None, :
                    ],
                ),
            },
            coords={"time": [t]},
        )

    return out


# --- bespoke view -----------------------------------------------------------


def _cross_section(data, time_index):
    """Poloidal cross-section: flux contours + separatrix + X/O-points.

    Renders via the shared shape-editor R-Z helpers (so it matches the waveform
    editor's diagram), degrading to a plain separatrix curve when those helpers
    are unavailable. Contours need the GGD psi grid; X/O-points need a filled
    contour_tree — both come from this profile's ``extract`` when present.
    """
    elements = []
    grid = data.dataset("derived/psi_grid")
    if grid is not None and rz_plot is not None:
        r = grid["r"].isel(time=time_index).values
        z = grid["z"].isel(time=time_index).values
        psi = grid["psi"].isel(time=time_index).values
        if len(r) and len(z) and len(psi):
            elements.append(rz_plot.contours(r, z, psi))

    sep = data.dataset("derived/separatrix")
    if sep is not None:
        r = sep["r"].isel(time=time_index).values
        z = sep["z"].isel(time=time_index).values
        if rz_plot is not None:
            elements.append(rz_plot.separatrix(r, z))
        else:
            elements.append(hv.Curve((r, z)).opts(color="red", line_width=3))

    xo = data.dataset("derived/xo_points")
    if xo is not None and rz_plot is not None:
        r = xo["r"].isel(time=time_index).values
        z = xo["z"].isel(time=time_index).values
        ctype = xo["critical_type"].isel(time=time_index).values
        x_pts = [(rr, zz) for rr, zz, c in zip(r, z, ctype) if c == 1]
        o_pts = [(rr, zz) for rr, zz, c in zip(r, z, ctype) if c in (0, 2)]
        elements.append(rz_plot.xo_points(x_pts, o_pts))

    if not elements:
        return hv.Curve(([], [])).opts(title="No equilibrium geometry")
    if rz_plot is not None:
        return rz_plot.plasma_overlay(elements).opts(responsive=True)
    return hv.Overlay(elements).opts(
        title="Poloidal flux",
        xlabel="r [m]",
        ylabel="z [m]",
        aspect="equal",
        responsive=True,
        show_legend=False,
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
    """Bespoke equilibrium overview at ``time_index`` (see module doc)."""
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

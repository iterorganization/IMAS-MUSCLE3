import runpy
from pathlib import Path

import imas
import numpy as np
import pytest


def _make_pds_equilibrium_slice(t, n_points=5, n_profile=10):
    """Create a single-time-slice equilibrium IDS with all fields required
    by pds.py."""
    eq = imas.IDSFactory("4.0.0").equilibrium()
    eq.ids_properties.homogeneous_time = 0
    eq.time = [t]
    eq.time_slice.resize(1)
    ts = eq.time_slice[0]
    ts.time = t

    ts.boundary.outline.r = np.linspace(1.5, 2.5, n_points)
    ts.boundary.outline.z = np.linspace(-1.0, 1.0, n_points)
    ts.boundary.psi = 1.0 + t * 0.1

    ts.ggd.resize(1)
    ts.ggd[0].r.resize(1)
    ts.ggd[0].z.resize(1)
    ts.ggd[0].psi.resize(1)
    ts.ggd[0].r[0].values = np.linspace(1, 5, n_points)
    ts.ggd[0].z[0].values = np.linspace(-2, 2, n_points)
    ts.ggd[0].psi[0].values = np.linspace(0, 1, n_points)

    ts.contour_tree.node.resize(2)
    ts.contour_tree.node[0].critical_type = 1  # X-point
    ts.contour_tree.node[0].r = 2.0
    ts.contour_tree.node[0].z = -1.5
    ts.contour_tree.node[1].critical_type = 0  # O-point
    ts.contour_tree.node[1].r = 3.0
    ts.contour_tree.node[1].z = 0.0

    ts.profiles_1d.f_df_dpsi = np.linspace(0, 1, n_profile)
    ts.profiles_1d.dpressure_dpsi = np.linspace(0, -1, n_profile)
    ts.profiles_1d.psi = np.linspace(0, 1, n_profile)

    ts.global_quantities.ip = 1e6 + t * 1e4
    ts.global_quantities.beta_tor = 0.05 + t * 0.01

    return eq


def _make_pds_pf_active_slice(t, n_coils=3):
    """Create a single-time-slice pf_active IDS with all fields required
    by pds.py."""
    pfa = imas.IDSFactory("4.0.0").pf_active()
    pfa.ids_properties.homogeneous_time = 0
    pfa.time = [t]
    pfa.coil.resize(n_coils)
    for i, coil in enumerate(pfa.coil):
        coil.name = f"Coil_{i}"
        coil.current.data = np.array([1000.0 * (i + 1) + t])
    return pfa


def test_pds_different_time_bases(monkeypatch):
    """Test that pds.py State handles equilibrium and pf_active IDSs
    with slightly different time bases without errors."""
    monkeypatch.setattr("panel.serve", lambda *args, **kwargs: None)

    current_dir = Path(__file__).parent
    pds_script_path = (
        current_dir / "../imas_muscle3/visualization/examples/pds/pds.py"
    ).resolve()
    if not pds_script_path.exists():
        pytest.fail(f"pds.py script not found at: {pds_script_path}")

    State = runpy.run_path(str(pds_script_path))["State"]
    state = State(md_dict={})

    eq_times = [0.0, 1.0, 2.0]
    pf_times = [0.05, 1.05, 2.05]

    for t in eq_times:
        state.extract(_make_pds_equilibrium_slice(t))

    for t in pf_times:
        state.extract(_make_pds_pf_active_slice(t))

    eq_data = state.data["equilibrium"]
    assert np.allclose(eq_data.time.values, eq_times)
    assert np.allclose(eq_data.ip.values, [1e6 + t * 1e4 for t in eq_times])
    assert np.allclose(
        eq_data.beta_tor.values, [0.05 + t * 0.01 for t in eq_times]
    )

    pf_data = state.data["pf_active"]
    assert np.allclose(pf_data.time.values, pf_times)
    assert pf_data.sizes["coil"] == 3

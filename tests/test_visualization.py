import runpy
import socket
from pathlib import Path

import imas
import numpy as np
import pytest
import ymmsl
from imas import DBEntry, ids_defs
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir
from libmuscle.pytest import MuscleTester

from conftest import slice_messages
from imas_muscle3.visualization.visualization_actor import VisualizationActor


def get_free_port() -> int:
    """Finds and returns an available port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_ymmsl_config(settings: dict) -> str:
    settings_str = "\n".join(f"  {k}: {v}" for k, v in settings.items())

    return f"""
ymmsl_version: v0.2
programs:
  visualization_component:
    ports:
      s: [equilibrium_in]
    executable: python
    args: -u -m imas_muscle3.actors.visualization_component
settings:
{settings_str}
"""


def test_visualization_actor(
    muscle3_tester: MuscleTester, equilibrium
) -> None:
    port = get_free_port()

    current_dir = Path(__file__).parent
    plot_script_path = (
        current_dir
        / "../imas_muscle3/visualization/examples/simple_1d_plot"
        / "simple_1d_plot.py"
    ).resolve()
    if not plot_script_path.exists():
        pytest.fail(f"Example plot script not found at: {plot_script_path}")

    settings = {
        "visualization_component.plot_file_path": str(plot_script_path),
        "visualization_component.port": port,
        "visualization_component.throttle_interval": 0,
        "visualization_component.keep_alive": False,
        "visualization_component.open_browser": False,
    }

    tester = muscle3_tester.start_implementation(
        create_ymmsl_config(settings), "visualization_component"
    )

    for msg in slice_messages(equilibrium, "equilibrium"):
        tester.send("equilibrium_in", msg)

    # No exception raised while sending means the actor consumed every
    # timeslice and shut its server down cleanly.


def create_full_ymmsl_config(settings: dict) -> str:
    """A source_component + visualization_component pair wired through a
    real Manager.
    """
    settings_str = "\n".join(f"  {k}: {v}" for k, v in settings.items())

    return f"""
ymmsl_version: v0.2
models:
  test_model:
    components:
      source_component:
        description: source component
        implementation: source_component
        ports:
          o_i: [equilibrium_out]
      visualization_component:
        description: visualization component
        implementation: visualization_component
        ports:
          s: [equilibrium_in]
    conduits:
      source_component.equilibrium_out: visualization_component.equilibrium_in
settings:
{settings_str}
programs:
  visualization_component:
    executable: python
    args: -u -m imas_muscle3.actors.visualization_component
  source_component:
    executable: python
    args: -u -m imas_muscle3.actors.source_component
resources:
  source_component:
    threads: 1
  visualization_component:
    threads: 1
"""


def run_and_check_for_error(
    tmp_path: Path, equilibrium, ymmsl_settings, expected_error
) -> None:
    """Run a source_component + visualization_component pair and check that
    a specific error ends up in the visualization actor's stderr log."""
    data_source_path = (tmp_path / "source_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(equilibrium)

    ymmsl_settings = {
        "source_component.source_uri": source_uri,
        **ymmsl_settings,
    }
    config = ymmsl.load(create_full_ymmsl_config(ymmsl_settings))
    run_dir = RunDir(tmp_path / "run")
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    assert not success

    log_file = run_dir.path / "instances/visualization_component/stderr.txt"
    assert log_file.exists()
    log_text = log_file.read_text()
    assert expected_error in log_text


def test_visualization_actor_no_plot_file(equilibrium, tmp_path) -> None:
    port = get_free_port()
    settings = {
        "visualization_component.plot_file_path": (
            "/path/to/non/existent/file.py"
        ),
        "visualization_component.port": port,
        "visualization_component.throttle_interval": 0,
        "visualization_component.keep_alive": False,
        "visualization_component.open_browser": False,
    }
    run_and_check_for_error(
        tmp_path, equilibrium, settings, "FileNotFoundError"
    )


def test_visualization_actor_missing_classes(equilibrium, tmp_path) -> None:
    script_path = tmp_path / "bad_plot.py"
    script_path.write_text("class NotState: pass\nclass NotPlotter: pass")
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must have a 'State' and a 'Plotter' class."
    run_and_check_for_error(tmp_path, equilibrium, settings, expected_error)


def test_visualization_actor_bad_state_inheritance(
    equilibrium, tmp_path
) -> None:
    script_path = tmp_path / "bad_inheritance.py"
    script_path.write_text(
        """
from imas_muscle3.visualization.base_plotter import BasePlotter
class State: pass  # Does not inherit from BaseState
class Plotter(BasePlotter): pass
"""
    )
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must inherit from BaseState"
    run_and_check_for_error(tmp_path, equilibrium, settings, expected_error)


def test_visualization_actor_bad_plotter_inheritance(
    equilibrium, tmp_path
) -> None:
    script_path = tmp_path / "bad_inheritance.py"
    script_path.write_text(
        """
from imas_muscle3.visualization.base_state import BaseState
class State(BaseState): pass
class Plotter: pass  # Does not inherit from BasePlotter
"""
    )
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must inherit from BasePlotter"
    run_and_check_for_error(tmp_path, equilibrium, settings, expected_error)


def test_state_data(equilibrium, monkeypatch):
    monkeypatch.setattr("panel.serve", lambda *args, **kwargs: None)
    current_dir = Path(__file__).parent
    plot_script_path = (
        current_dir
        / "../imas_muscle3/visualization/examples/simple_1d_plot"
        / "simple_1d_plot.py"
    ).resolve()
    if not plot_script_path.exists():
        pytest.fail(f"Example plot script not found at: {plot_script_path}")
    actor = VisualizationActor(
        plot_file_path=str(plot_script_path),
        port=1234,
        md_dict={},
        open_browser_on_start=False,
    )

    # Extract each time slice separately
    with DBEntry("imas:memory?path=/", "w") as db:
        db.put(equilibrium)
        for t in equilibrium.time:
            single_slice_ids = db.get_slice(
                "equilibrium", t, ids_defs.CLOSEST_INTERP
            )
            actor.state.extract(single_slice_ids)

    state_data = actor.plotter._state.data["equilibrium"]
    expected_times = equilibrium.time
    expected_ips = [ts.global_quantities.ip for ts in equilibrium.time_slice]
    assert np.all(state_data["time"] == expected_times)
    assert np.all(state_data["ip"] == expected_ips)


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

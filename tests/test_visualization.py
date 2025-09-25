import multiprocessing
import socket
from pathlib import Path

import numpy as np
import pytest
import ymmsl
from imas import DBEntry, ids_defs
from libmuscle.manager.manager import Manager
from libmuscle.manager.run_dir import RunDir

from imas_muscle3.visualization.examples.simple_1d_plot.simple_1d_plot import (
    Plotter,
    State,
)
from imas_muscle3.visualization.visualization_actor import VisualizationActor

"""Force 'spawn' start method to avoid deadlocks with pytest."""
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


def get_free_port() -> int:
    """Finds and returns an available port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_ymmsl_config(settings: dict) -> str:
    settings_str = "\n".join(f"  {k}: {v}" for k, v in settings.items())

    return f"""
ymmsl_version: v0.1
model:
  name: test_model
  components:
    source_component:
      implementation: source_component
      ports:
        o_i: [equilibrium_out]
    visualization_component:
      implementation: visualization_component
      ports:
        s: [equilibrium_in]
  conduits:
    source_component.equilibrium_out: visualization_component.equilibrium_in
settings:
{settings_str}
implementations:
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


def test_visualization_actor(tmpdir, equilibrium):
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(equilibrium)

    port = get_free_port()
    tmppath = Path(str(tmpdir))

    current_dir = Path(__file__).parent
    plot_script_path = (
        current_dir
        / "../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py"
    ).resolve()
    if not plot_script_path.exists():
        pytest.fail(f"Example plot script not found at: {plot_script_path}")

    settings = {
        "source_component.source_uri": source_uri,
        "visualization_component.plot_file_path": str(plot_script_path),
        "visualization_component.port": port,
        "visualization_component.throttle_interval": 0,
        "visualization_component.keep_alive": False,
        "visualization_component.open_browser": False,
    }

    ymmsl_text = create_ymmsl_config(settings)
    config = ymmsl.load(ymmsl_text)
    run_dir = RunDir(tmppath / "run")
    manager = Manager(config, run_dir)
    manager.start_instances()

    success = manager.wait()
    assert success


def run_and_check_for_error(tmpdir, equilibrium, ymmsl_settings, expected_error):
    """Helper function to run a simulation and check for a specific error."""
    data_source_path = (Path(tmpdir) / "source_component_data").absolute()
    source_uri = f"imas:hdf5?path={data_source_path}"
    with DBEntry(source_uri, "w") as entry:
        entry.put(equilibrium)

    tmppath = Path(str(tmpdir))
    ymmsl_text = create_ymmsl_config(ymmsl_settings)
    config = ymmsl.load(ymmsl_text)
    run_dir = RunDir(tmppath / "run")
    manager = Manager(config, run_dir)
    manager.start_instances()
    success = manager.wait()

    assert not success

    log_file = run_dir.path / "instances/visualization_component/stderr.txt"
    assert log_file.exists()
    log_text = log_file.read_text()
    assert expected_error in log_text


def test_visualization_actor_no_plot_file(tmpdir, equilibrium):
    settings = {
        "visualization_component.plot_file_path": "/path/to/non/existent/file.py"
    }
    run_and_check_for_error(tmpdir, equilibrium, settings, "FileNotFoundError")


def test_visualization_actor_missing_classes(tmpdir, equilibrium, tmp_path):
    script_path = tmp_path / "bad_plot.py"
    script_path.write_text("class NotState: pass\nclass NotPlotter: pass")
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must define a 'State' class and a 'Plotter' class"
    run_and_check_for_error(tmpdir, equilibrium, settings, expected_error)


def test_visualization_actor_bad_state_inheritance(tmpdir, equilibrium, tmp_path):
    script_path = tmp_path / "bad_inheritance.py"
    script_path.write_text(
        """
from imas_muscle3.visualization.base import BasePlotter
class State: pass  # Does not inherit from BaseState
class Plotter(BasePlotter): pass
"""
    )
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must inherit from BaseState"
    run_and_check_for_error(tmpdir, equilibrium, settings, expected_error)


def test_visualization_actor_bad_plotter_inheritance(tmpdir, equilibrium, tmp_path):
    script_path = tmp_path / "bad_inheritance.py"
    script_path.write_text(
        """
from imas_muscle3.visualization.base import BaseState
class State(BaseState): pass
class Plotter: pass  # Does not inherit from BasePlotter
"""
    )
    settings = {"visualization_component.plot_file_path": str(script_path)}
    expected_error = "must inherit from BasePlotter"
    run_and_check_for_error(tmpdir, equilibrium, settings, expected_error)


def test_state_data(equilibrium, monkeypatch):
    monkeypatch.setattr("panel.serve", lambda *args, **kwargs: None)
    current_dir = Path(__file__).parent
    plot_script_path = (
        current_dir
        / "../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py"
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
            single_slice_ids = db.get_slice("equilibrium", t, ids_defs.CLOSEST_INTERP)
            actor.state.extract(single_slice_ids)

    state_data = actor.plotter._state.data["equilibrium"]
    expected_times = equilibrium.time
    expected_ips = [ts.global_quantities.ip for ts in equilibrium.time_slice]
    assert np.all(state_data["time"] == expected_times)
    assert np.all(state_data["ip"] == expected_ips)

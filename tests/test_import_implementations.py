from importlib.metadata import version
from pathlib import Path

import pytest
import yaml
import ymmsl

ymmsl_version_tuple = tuple(map(int, version("ymmsl").split(".")[:2]))
skip_import_tests = pytest.mark.skipif(
    ymmsl_version_tuple < (0, 16),
    reason="Entry points not implemented in yMMSL",
)

components = [
    "source_component",
    "sink_component",
    "sink_source_component",
    "olc_component",
    "accumulator_component",
    "iterator_component",
    "passthrough_component",
    "recorder_component",
    "visualization_component",
]


def test_all_actor_modules_are_registered() -> None:
    """Every module in imas_muscle3.actors is exposed as a ymmsl program."""
    import imas_muscle3.actors

    modules = {
        path.stem
        for path in Path(imas_muscle3.actors.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    }
    registered = set(yaml.safe_load(imas_muscle3.actors.ACTORS)["programs"])
    assert modules == registered
    # the parametrized test below covers exactly these
    assert modules == set(components)


@skip_import_tests
@pytest.mark.parametrize("component_name", components)
def test_import_component(component_name: str) -> None:
    # local import to avoid ImportError for older ymmsl versions
    from ymmsl.v0_2 import Configuration, Reference, resolve

    config = ymmsl.load_as(
        Configuration,
        f"""
ymmsl_version: v0.2
imports:
- from imas_muscle3 import implementation {component_name}
models:
  test:
    components:
      test_component:
        ports: {{}}
        description: Test component
        implementation: {component_name}
resources:
  test_component:
    threads: 1
""",
    )
    resolve(Reference([]), config)
    config.check_consistent()

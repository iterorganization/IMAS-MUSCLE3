from pathlib import Path

import pytest
import ymmsl
from ymmsl.v0_2 import Configuration, Reference, resolve

import imas_muscle3.actors

# parametrize based on modules on disk, so a new actor is covered by
# test_import_component as soon as it is added
components = sorted(
    path.stem
    for path in Path(imas_muscle3.actors.__file__).parent.glob("*.py")
    if not path.stem.startswith("_")
)


def test_all_actor_modules_are_registered() -> None:
    """Modules and exposed programs match, in both directions."""
    registered_programs = ymmsl.load_as(
        Configuration, imas_muscle3.actors.ACTORS
    ).programs
    assert set(components) == registered_programs.keys()


@pytest.mark.parametrize("component_name", components)
def test_import_component(component_name: str) -> None:
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

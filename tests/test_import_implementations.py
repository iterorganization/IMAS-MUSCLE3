from importlib.metadata import version

import ymmsl
import pytest

ymmsl_version_tuple = tuple(map(int, version("ymmsl").split(".")[:2]))
skip_import_tests = pytest.mark.skipif(
    ymmsl_version_tuple < (0, 16), reason="Entry points not implemented in yMMSL"
)

components = [
    "source_component",
    "sink_component",
    "sink_source_component",
    "olc_component",
    "accumulator_component",
    "visualization_component",
]


@skip_import_tests
@pytest.mark.parametrize("component_name", components)
def test_import_component(component_name: str) -> None:
    # local import to avoid ImportError for older ymmsl versions
    from ymmsl.v0_2 import Configuration, resolve, Reference

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

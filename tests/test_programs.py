import pkgutil
from importlib.metadata import entry_points

import ymmsl
from ymmsl import v0_2

import imas_muscle3.actors


def test_entry_point():
    eps = entry_points(group="ymmsl.module", name="imas_muscle3.actors")
    assert len(eps) == 1
    config = ymmsl.load_as(v0_2.Configuration, next(iter(eps)).load())

    actors = {
        module.name
        for module in pkgutil.iter_modules(imas_muscle3.actors.__path__)
        if module.name.endswith("_component")
    }
    assert set(map(str, config.programs)) == actors
    for name, program in config.programs.items():
        assert program.args == [f"-u -m imas_muscle3.actors.{name}"]

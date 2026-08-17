from typing import Any

import pytest
from imas import DBEntry
from libmuscle import Message
from libmuscle.pytest import MuscleTester

from conftest import deserialize, receive_all
from imas_muscle3.actors.iterator_component import (
    IteratorSettings,
    determine_t_array,
)

SINGLE_IDS_CONFIG = """
ymmsl_version: v0.2
programs:
  iterator:
    ports:
      f_init: [equilibrium_in]
      o_i: [equilibrium_out]
    executable: python
    args: -u -m imas_muscle3.actors.iterator_component
"""


class _FakeInstance:
    """Minimal stand-in for libmuscle's Instance, just enough to drive
    IteratorSettings.from_instance without starting a real actor."""

    def __init__(self, settings: dict) -> None:
        self._settings = settings

    def get_setting(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name in self._settings:
            return self._settings[name]
        if "default" in kwargs:
            return kwargs["default"]
        raise KeyError(name)


def _start(muscle3_tester: MuscleTester, setting_line: str) -> MuscleTester:
    config = f"""
ymmsl_version: v0.2
programs:
  iterator:
    ports:
      f_init: [core_profiles_in, equilibrium_in]
      o_i: [core_profiles_out, equilibrium_out]
    executable: python
    args: -u -m imas_muscle3.actors.iterator_component
settings:
  {setting_line}
"""
    return muscle3_tester.start_implementation(config, "iterator")


def test_time_source_ids(
    muscle3_tester: MuscleTester, core_profiles, equilibrium
) -> None:
    """Uses core_profiles' own time array as the reference timeslices,
    applied to every connected IDS."""
    tester = _start(muscle3_tester, "iterator.time_source_ids: core_profiles")
    tester.send(
        "core_profiles_in", Message(0.0, data=core_profiles.serialize())
    )
    tester.send("equilibrium_in", Message(0.0, data=equilibrium.serialize()))

    cp_messages = receive_all(tester, "core_profiles_out")
    eq_messages = receive_all(tester, "equilibrium_out")
    assert [m.timestamp for m in cp_messages] == list(core_profiles.time)
    assert [m.timestamp for m in eq_messages] == list(core_profiles.time)
    for msg in cp_messages:
        result = deserialize("core_profiles", msg.data)
        assert list(result.time) == [msg.timestamp]


def test_n_timeslices(
    muscle3_tester: MuscleTester, core_profiles, equilibrium
) -> None:
    """core_profiles.time spans [0.0, 2.0]; 3 evenly spaced slices should
    land on 0.0, 1.0, 2.0."""
    tester = _start(
        muscle3_tester,
        "iterator.time_source_ids: core_profiles\n  iterator.n_timeslices: 3",
    )
    tester.send(
        "core_profiles_in", Message(0.0, data=core_profiles.serialize())
    )
    tester.send("equilibrium_in", Message(0.0, data=equilibrium.serialize()))

    cp_messages = receive_all(tester, "core_profiles_out")
    assert [m.timestamp for m in cp_messages] == [0.0, 1.0, 2.0]


def test_auto_infers_time_source_ids_with_single_connected_ids(
    muscle3_tester: MuscleTester, equilibrium
) -> None:
    tester = muscle3_tester.start_implementation(SINGLE_IDS_CONFIG, "iterator")
    tester.send("equilibrium_in", Message(0.0, data=equilibrium.serialize()))

    messages = receive_all(tester, "equilibrium_out")
    assert [m.timestamp for m in messages] == list(equilibrium.time)


def test_determine_t_array_infers_single_connected_ids(core_profiles) -> None:
    settings = IteratorSettings.from_instance(_FakeInstance({}))
    with DBEntry("imas:memory?path=/", "w") as db:
        db.put(core_profiles)
        t_array = determine_t_array(db, ["core_profiles_in"], settings)
    assert t_array == list(core_profiles.time)


def test_determine_t_array_requires_time_source_ids_for_multiple(
    core_profiles, equilibrium
) -> None:
    settings = IteratorSettings.from_instance(_FakeInstance({}))
    with DBEntry("imas:memory?path=/", "w") as db:
        db.put(core_profiles)
        db.put(equilibrium)
        with pytest.raises(ValueError):
            determine_t_array(
                db, ["core_profiles_in", "equilibrium_in"], settings
            )


def test_time_source_ids_is_optional() -> None:
    settings = IteratorSettings.from_instance(_FakeInstance({}))
    assert settings.time_source_ids is None


def test_n_timeslices_is_optional() -> None:
    settings = IteratorSettings.from_instance(
        _FakeInstance({"time_source_ids": "core_profiles"})
    )
    assert settings.n_timeslices is None


def test_n_timeslices_can_be_combined_with_time_source_ids() -> None:
    settings = IteratorSettings.from_instance(
        _FakeInstance({"time_source_ids": "core_profiles", "n_timeslices": 3})
    )
    assert settings.time_source_ids == "core_profiles"
    assert settings.n_timeslices == 3

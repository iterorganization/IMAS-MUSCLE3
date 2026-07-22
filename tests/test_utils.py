from unittest.mock import Mock

import numpy
import pytest
from imas_core.exception import ImasCoreBackendException
from ymmsl.v0_2 import Operator

from imas_muscle3.data_sink_source import get_sink_db_entry
from imas_muscle3.utils import (
    get_port_list,
    get_setting_optional,
    ids_from_message,
    ids_name_from_port,
    increment_suffix,
)


def test_increment_suffix():
    assert (
        increment_suffix("imas:hdf5?path=my_string")
        == "imas:hdf5?path=my_string_1"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string_1")
        == "imas:hdf5?path=my_string_2"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string_99")
        == "imas:hdf5?path=my_string_100"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string_09")
        == "imas:hdf5?path=my_string_10"
    )
    assert (
        increment_suffix("imas:hdf5?path=my/string")
        == "imas:hdf5?path=my/string_1"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_1_string")
        == "imas:hdf5?path=my_1_string_1"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_1_string_2")
        == "imas:hdf5?path=my_1_string_3"
    )
    assert increment_suffix("imas:hdf5?path=9") == "imas:hdf5?path=9_1"
    assert increment_suffix("imas:hdf5?path=_9") == "imas:hdf5?path=_10"
    assert (
        increment_suffix("imas:hdf5?path=my_string&hdf5_debug=true")
        == "imas:hdf5?path=my_string_1&hdf5_debug=true"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string;hdf5_debug=true")
        == "imas:hdf5?path=my_string_1&hdf5_debug=true"
    )


@pytest.mark.parametrize(
    "backend",
    [
        # ('mdsplus'), no access to mdsplus on github CI
        ("hdf5"),
    ],
)
def test_get_sink_db_entry(tmp_path, backend):
    path = tmp_path / "init"
    path_inc = tmp_path / "init_1"
    uri = f"imas:{backend}?path={path};hdf5_debug=yes"
    assert not path.exists()
    assert not path_inc.exists()

    # check 'x' works
    get_sink_db_entry(uri, "x", True)
    assert path.exists()
    assert not path_inc.exists()

    # check 'w' overwrites
    get_sink_db_entry(uri, "w", True)
    assert not path_inc.exists()

    # check if error is thrown when setting disabled
    with pytest.raises(ImasCoreBackendException):
        get_sink_db_entry(uri, "x", False)
    assert not path_inc.exists()

    # check 'x' adds number
    get_sink_db_entry(uri, "x", True)
    assert path.exists()
    assert path_inc.exists()

    # check if error is thrown when using non-path uri
    legacy_uri = (
        f"imas:{backend}?pulse=123;run=2;user={tmp_path};"
        "database=ITER;version=3"
    )
    get_sink_db_entry(legacy_uri, "x", False)
    with pytest.raises(ImasCoreBackendException):
        get_sink_db_entry(legacy_uri, "x", False)


def test_get_setting_optional_returns_value_when_present():
    instance = Mock()
    instance.get_setting.return_value = "my_value"
    assert get_setting_optional(instance, "some_setting") == "my_value"


def test_get_setting_optional_returns_none_when_missing_and_no_default():
    instance = Mock()
    instance.get_setting.side_effect = KeyError("some_setting")
    assert get_setting_optional(instance, "some_setting") is None


def test_get_setting_optional_returns_default_when_missing():
    instance = Mock()
    instance.get_setting.side_effect = KeyError("some_setting")
    assert (
        get_setting_optional(instance, "some_setting", default="fallback")
        == "fallback"
    )


def test_get_port_list_filters_to_connected_and_sorts():
    instance = Mock()
    instance.list_ports.return_value = {
        Operator.S: ["c_in", "a_in", "b_in"],
    }
    instance.is_connected.side_effect = lambda port: port != "b_in"
    assert get_port_list(instance, Operator.S) == ["a_in", "c_in"]


def test_get_port_list_returns_empty_for_unknown_operator():
    instance = Mock()
    instance.list_ports.return_value = {Operator.S: ["a_in"]}
    instance.is_connected.return_value = True
    assert get_port_list(instance, Operator.O_F) == []


def test_ids_name_from_port_strips_in_suffix():
    assert ids_name_from_port("core_profiles_in") == "core_profiles"


def test_ids_name_from_port_without_in_suffix():
    assert ids_name_from_port("equilibrium") == "equilibrium"


def test_ids_name_from_port_rejects_unknown_ids():
    with pytest.raises(ValueError):
        ids_name_from_port("not_a_real_ids")


def test_ids_from_message_roundtrip(core_profiles):
    data = core_profiles.serialize()
    ids = ids_from_message("core_profiles", data)
    assert numpy.array_equal(ids.time, core_profiles.time)

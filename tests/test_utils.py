import pytest

from imas_muscle3.utils import increment_suffix
from imas_muscle3.data_sink_source import get_sink_db_entry
from imas_core.exception import ImasCoreBackendException


def test_increment_suffix():
    assert increment_suffix("imas:hdf5?path=my_string") == "imas:hdf5?path=my_string_1"
    assert (
        increment_suffix("imas:hdf5?path=my_string_1") == "imas:hdf5?path=my_string_2"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string_99")
        == "imas:hdf5?path=my_string_100"
    )
    assert (
        increment_suffix("imas:hdf5?path=my_string_09") == "imas:hdf5?path=my_string_10"
    )
    assert increment_suffix("imas:hdf5?path=my/string") == "imas:hdf5?path=my/string_1"
    assert (
        increment_suffix("imas:hdf5?path=my_1_string") == "imas:hdf5?path=my_1_string_1"
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
        f"imas:{backend}?pulse=123;run=2;user={tmp_path};database=ITER;version=3"
    )
    get_sink_db_entry(legacy_uri, "x", False)
    with pytest.raises(ImasCoreBackendException):
        get_sink_db_entry(legacy_uri, "x", False)

import imas
import pytest
from libmuscle import Message
from libmuscle.pytest import MuscleTester

from conftest import deserialize
from imas_muscle3.actors.merger_component import (
    active_ids_names,
    check_time,
    merge,
)

DD_VERSION = "4.1.1"

CONFIG = """
ymmsl_version: v0.2
programs:
  merger:
    ports:
      f_init: [{f_init_ports}]
      o_f: [{o_f_ports}]
    executable: python
    args: -u -m imas_muscle3.actors.merger_component
"""


def _start(
    muscle3_tester: MuscleTester, f_init: str, o_f: str
) -> MuscleTester:
    return muscle3_tester.start_implementation(
        CONFIG.format(f_init_ports=f_init, o_f_ports=o_f), "merger"
    )


def _equilibrium(times, homogeneous=True):
    """A homogeneous equilibrium with one filled node per time slice."""
    eq = imas.IDSFactory(DD_VERSION).equilibrium()
    eq.ids_properties.homogeneous_time = 1 if homogeneous else 0
    eq.time = times
    eq.time_slice.resize(len(times))
    for i, t in enumerate(times):
        eq.time_slice[i].time = t
    return eq


def _core_profiles(times):
    cp = imas.IDSFactory(DD_VERSION).core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.time = times
    return cp


def _merge(tester, base, overlay, ids_name="equilibrium"):
    """Send a base and an overlay, and return the merged IDS."""
    tester.send(f"{ids_name}_base", Message(0.0, data=base.serialize()))
    tester.send(f"{ids_name}_overlay", Message(0.0, data=overlay.serialize()))
    reply = tester.receive(f"{ids_name}_out")
    return deserialize(ids_name, reply.data)


def test_overlay_wins_and_base_survives(
    muscle3_tester: MuscleTester,
) -> None:
    """Every filled overlay node is written over the base; everything the
    overlay is silent about keeps the base's value."""
    tester = _start(
        muscle3_tester,
        "equilibrium_base, equilibrium_overlay",
        "equilibrium_out",
    )
    times = [0.0, 1.0, 2.0]

    base = _equilibrium(times)
    base.vacuum_toroidal_field.r0 = 6.2
    base.vacuum_toroidal_field.b0 = [-2.0, -2.0, -2.0]
    for i, ts in enumerate(base.time_slice):
        ts.global_quantities.ip = 1e6 + i * 1e5
        ts.global_quantities.psi_boundary = 10.0 + i
        ts.profiles_1d.psi = [0.0, 1.0, 2.0]

    overlay = _equilibrium(times)
    overlay.vacuum_toroidal_field.b0 = [-2.65, -2.65, -2.65]
    for ts in overlay.time_slice:
        ts.global_quantities.ip = -3e6

    merged = _merge(tester, base, overlay)

    # Overlaid.
    assert [float(t.global_quantities.ip) for t in merged.time_slice] == [
        -3e6
    ] * 3
    assert list(merged.vacuum_toroidal_field.b0) == [-2.65] * 3
    # Untouched by the overlay, so the base's values survive.
    assert float(merged.vacuum_toroidal_field.r0) == 6.2
    assert [
        float(t.global_quantities.psi_boundary) for t in merged.time_slice
    ] == [10.0, 11.0, 12.0]
    assert all(
        list(t.profiles_1d.psi) == [0.0, 1.0, 2.0] for t in merged.time_slice
    )


def test_several_ids_names(muscle3_tester: MuscleTester) -> None:
    """One instance carries any number of IDSs, merged independently."""
    tester = _start(
        muscle3_tester,
        "equilibrium_base, equilibrium_overlay, "
        "core_profiles_base, core_profiles_overlay",
        "equilibrium_out, core_profiles_out",
    )
    times = [0.0, 1.0]

    eq_base = _equilibrium(times)
    eq_base.vacuum_toroidal_field.r0 = 6.2
    eq_overlay = _equilibrium(times)
    eq_overlay.vacuum_toroidal_field.b0 = [-2.65, -2.65]

    cp_base = _core_profiles(times)
    cp_base.global_quantities.v_loop = [1.0, 2.0]
    cp_overlay = _core_profiles(times)
    cp_overlay.global_quantities.ip = [-3e6, -3e6]

    for name, base, overlay in (
        ("equilibrium", eq_base, eq_overlay),
        ("core_profiles", cp_base, cp_overlay),
    ):
        tester.send(f"{name}_base", Message(0.0, data=base.serialize()))
        tester.send(f"{name}_overlay", Message(0.0, data=overlay.serialize()))

    eq = deserialize("equilibrium", tester.receive("equilibrium_out").data)
    cp = deserialize("core_profiles", tester.receive("core_profiles_out").data)

    assert float(eq.vacuum_toroidal_field.r0) == 6.2
    assert list(eq.vacuum_toroidal_field.b0) == [-2.65, -2.65]
    assert list(cp.global_quantities.v_loop) == [1.0, 2.0]
    assert list(cp.global_quantities.ip) == [-3e6, -3e6]


def test_array_of_structure_is_merged_per_element(
    muscle3_tester: MuscleTester,
) -> None:
    """An overlay shorter than the base leaves the base's extra elements
    alone, rather than truncating them."""
    tester = _start(
        muscle3_tester, "pf_active_base, pf_active_overlay", "pf_active_out"
    )
    factory = imas.IDSFactory(DD_VERSION)

    base = factory.pf_active()
    base.ids_properties.homogeneous_time = 1
    base.time = [0.0, 1.0]
    base.coil.resize(3)
    for i, coil in enumerate(base.coil):
        coil.name = f"coil{i}"
        coil.current.data = [0.0, 0.0]

    overlay = factory.pf_active()
    overlay.ids_properties.homogeneous_time = 1
    overlay.time = [0.0, 1.0]
    overlay.coil.resize(2)
    for i, coil in enumerate(overlay.coil):
        coil.current.data = [float(i), float(i)]

    merged = _merge(tester, base, overlay, "pf_active")

    assert len(merged.coil) == 3
    assert [str(c.name) for c in merged.coil] == ["coil0", "coil1", "coil2"]
    assert [list(c.current.data) for c in merged.coil] == [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]


class _FakeInstance:
    """Minimal stand-in for libmuscle's Instance, just enough to drive
    active_ids_names without starting a real actor."""

    def __init__(self, connected: set) -> None:
        self._connected = connected

    def is_connected(self, port_name: str) -> bool:
        return port_name in self._connected


@pytest.mark.parametrize(
    "base, overlay, expected",
    [
        pytest.param(
            [0.0, 1.0, 2.0],
            [0.0, 1.0],
            "same time base",
            id="different_length",
        ),
        pytest.param(
            [0.0, 1.0, 2.0],
            [0.0, 1.5, 2.0],
            "differ from step 1",
            id="different_values",
        ),
        pytest.param([], [0.0, 1.0], "empty root /time", id="empty_base"),
        pytest.param([0.0, 1.0], [], "empty root /time", id="empty_overlay"),
    ],
)
def test_check_time_refuses_mismatched_time(base, overlay, expected) -> None:
    """The merge writes node onto node with no regard for time, so anything
    but one shared time base is refused."""
    with pytest.raises(RuntimeError, match=expected):
        check_time("equilibrium", _equilibrium(base), _equilibrium(overlay))


@pytest.mark.parametrize("side", ["base", "overlay"], ids=["base", "overlay"])
def test_check_time_refuses_heterogeneous(side) -> None:
    """A heterogeneous IDS has no single root /time to compare, even when it
    happens to be populated."""
    times = [0.0, 1.0]
    ids = {
        "base": (_equilibrium(times, homogeneous=False), _equilibrium(times)),
        "overlay": (
            _equilibrium(times),
            _equilibrium(times, homogeneous=False),
        ),
    }[side]
    with pytest.raises(RuntimeError, match="not in homogeneous time mode"):
        check_time("equilibrium", *ids)


def test_check_time_accepts_one_shared_time_base() -> None:
    times = [0.0, 1.0, 2.0]
    check_time("equilibrium", _equilibrium(times), _equilibrium(times))


def test_merge_refuses_overlay_longer_than_base() -> None:
    """Growing the base would mean inventing the elements it does not have,
    so the merge stops instead."""
    factory = imas.IDSFactory(DD_VERSION)
    base = factory.pf_active()
    base.coil.resize(2)
    overlay = factory.pf_active()
    overlay.coil.resize(5)
    for coil in overlay.coil:
        coil.current.data = [0.0]

    with pytest.raises(RuntimeError, match="5 element"):
        merge(base, overlay, "pf_active")


def test_active_ids_names_refuses_half_wired_ids() -> None:
    """Receiving on an unconnected port blocks forever, so a partially wired
    IDS fails at startup instead."""
    instance = _FakeInstance({"equilibrium_base", "equilibrium_out"})
    with pytest.raises(RuntimeError, match="half wired"):
        active_ids_names(instance)


def test_active_ids_names_refuses_nothing_connected() -> None:
    with pytest.raises(RuntimeError, match="nothing is connected"):
        active_ids_names(_FakeInstance(set()))


def test_active_ids_names_lists_fully_wired_ids() -> None:
    instance = _FakeInstance(
        {
            "equilibrium_base",
            "equilibrium_overlay",
            "equilibrium_out",
            "core_profiles_base",
            "core_profiles_overlay",
            "core_profiles_out",
        }
    )
    assert sorted(active_ids_names(instance)) == [
        "core_profiles",
        "equilibrium",
    ]

import imas
import pytest
from libmuscle import Message
from libmuscle.pytest import MuscleTester

from conftest import slice_messages
from imas_muscle3.actors.passthrough_component import _validate_ports

CONFIG = """
ymmsl_version: v0.2
programs:
  passthrough:
    ports:
      f_init: [{f_init_ports}]
      s: [{s_ports}]
      o_f: [{o_f_ports}]
      o_i: [{o_i_ports}]
    executable: python
    args: -u -m imas_muscle3.actors.passthrough_component
"""

SINGLE_SLICE = [0.0]
MULTI_SLICE = [0.0, 1.0, 2.0]


def _start(muscle3_tester: MuscleTester, **ports) -> MuscleTester:
    return muscle3_tester.start_implementation(
        CONFIG.format(
            f_init_ports=ports.get("f_init", ""),
            s_ports=ports.get("s", ""),
            o_f_ports=ports.get("o_f", ""),
            o_i_ports=ports.get("o_i", ""),
        ),
        "passthrough",
    )


def _equilibrium(times):
    eq = imas.IDSFactory("4.0.0").equilibrium()
    eq.ids_properties.homogeneous_time = 0
    eq.time = times
    eq.time_slice.resize(len(times))
    for i, t in enumerate(times):
        eq.time_slice[i].time = t
        eq.time_slice[i].global_quantities.ip = 1e6 + i * 1e5
    return eq


@pytest.mark.parametrize(
    "times", [SINGLE_SLICE, MULTI_SLICE], ids=["single_slice", "multi_slice"]
)
def test_f_init_forwarded_to_o_f(muscle3_tester: MuscleTester, times) -> None:
    tester = _start(
        muscle3_tester, f_init="equilibrium_in_f", o_f="equilibrium_out_f"
    )
    sent = Message(0.0, data=_equilibrium(times).serialize())
    tester.send("equilibrium_in_f", sent)
    reply = tester.receive("equilibrium_out_f")
    assert reply.data == sent.data


@pytest.mark.parametrize(
    "times", [SINGLE_SLICE, MULTI_SLICE], ids=["single_slice", "multi_slice"]
)
def test_s_forwarded_to_o_i(muscle3_tester: MuscleTester, times) -> None:
    """Every S timeslice is forwarded in order. For multiple slices this
    also checks that the actor keeps looping on S until the last slice
    (next_timestamp=None), rather than stopping after the first."""
    tester = _start(
        muscle3_tester, s="equilibrium_in_s", o_i="equilibrium_out_i"
    )
    sent = slice_messages(_equilibrium(times), "equilibrium")
    for msg in sent:
        tester.send("equilibrium_in_s", msg)
    received = [tester.receive("equilibrium_out_i") for _ in sent]
    assert [m.timestamp for m in received] == [m.timestamp for m in sent]
    assert [m.data for m in received] == [m.data for m in sent]


def test_raises_for_output_without_matching_input() -> None:
    with pytest.raises(RuntimeError):
        _validate_ports(
            in_f=set(), in_s=set(), out_f={"equilibrium"}, out_i=set()
        )


def test_does_not_raise_for_input_without_matching_output() -> None:
    _validate_ports(in_f={"equilibrium"}, in_s=set(), out_f=set(), out_i=set())

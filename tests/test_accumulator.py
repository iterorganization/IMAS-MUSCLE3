import imas
import pytest
from libmuscle import Message
from libmuscle.pytest import MuscleTester

from conftest import slice_messages

CONFIG = """
ymmsl_version: v0.2
programs:
  accumulator:
    ports:
      s: [{s_ports}]
      o_f: [core_profiles_out, equilibrium_out]
    executable: python
    args: -u -m imas_muscle3.actors.accumulator_component
"""


@pytest.mark.parametrize("use_t_next", [True, False])
def test_accumulator(
    muscle3_tester: MuscleTester,
    core_profiles,
    equilibrium,
    use_t_next: bool,
) -> None:
    # give the second port a different number of timeslices than
    # core_profiles, to check that each IDS's own timeslice count is
    # tracked independently.
    equilibrium.time = [0.0, 0.5, 1.0, 1.5, 2.0]
    equilibrium.time_slice.resize(len(equilibrium.time))
    for i, t in enumerate(equilibrium.time):
        equilibrium.time_slice[i].time = t
        equilibrium.time_slice[i].global_quantities.ip = 1e6 + i * 1e5

    s_ports = "core_profiles_in, equilibrium_in"
    if use_t_next:
        s_ports += ", t_next"
    tester = muscle3_tester.start_implementation(
        CONFIG.format(s_ports=s_ports), "accumulator"
    )

    cp_messages = slice_messages(core_profiles, "core_profiles")
    eq_messages = slice_messages(equilibrium, "equilibrium")
    n_steps = max(len(cp_messages), len(eq_messages))

    for i in range(n_steps):
        if i < len(cp_messages):
            tester.send("core_profiles_in", cp_messages[i])
        if i < len(eq_messages):
            tester.send("equilibrium_in", eq_messages[i])
        if use_t_next:
            next_t = float(i + 1) if i + 1 < n_steps else None
            tester.send("t_next", Message(float(i), next_timestamp=next_t))

    cp_reply = tester.receive("core_profiles_out")
    cp_result = imas.IDSFactory("4.0.0").core_profiles()
    cp_result.deserialize(cp_reply.data)
    assert all(cp_result.time == core_profiles.time)

    eq_reply = tester.receive("equilibrium_out")
    eq_result = imas.IDSFactory("4.0.0").equilibrium()
    eq_result.deserialize(eq_reply.data)
    assert all(eq_result.time == equilibrium.time)

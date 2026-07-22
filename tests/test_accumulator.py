from typing import List

import imas
import pytest
from imas import DBEntry
from imas.ids_defs import CLOSEST_INTERP
from libmuscle import Message
from libmuscle.pytest import MuscleTester

CONFIG = """
ymmsl_version: v0.2
programs:
  accumulator:
    ports:
      s: [{s_ports}]
      o_f: [core_profiles_out]
    executable: python
    args: -u -m imas_muscle3.actors.accumulator_component
"""


def _slice_messages(
    ids: imas.ids_toplevel.IDSToplevel, ids_name: str
) -> List[Message]:
    """Build one Message per timeslice of `ids`, chaining next_timestamp the
    way a real source component would."""
    messages = []
    times = list(ids.time)
    with DBEntry("imas:memory?path=/", "w") as db:
        db.put(ids)
        for i, t in enumerate(times):
            next_t = times[i + 1] if i + 1 < len(times) else None
            data = db.get_slice(ids_name, t, CLOSEST_INTERP).serialize()
            messages.append(Message(t, data=data, next_timestamp=next_t))
    return messages


@pytest.mark.parametrize("use_t_next", [True, False])
def test_accumulator(
    muscle3_tester: MuscleTester, core_profiles, use_t_next: bool
) -> None:
    s_ports = "core_profiles_in, t_next" if use_t_next else "core_profiles_in"
    tester = muscle3_tester.start_implementation(
        CONFIG.format(s_ports=s_ports), "accumulator"
    )

    for msg in _slice_messages(core_profiles, "core_profiles"):
        tester.send("core_profiles_in", msg)
        if use_t_next:
            tester.send(
                "t_next",
                Message(msg.timestamp, next_timestamp=msg.next_timestamp),
            )

    reply = tester.receive("core_profiles_out")
    result = imas.IDSFactory("4.0.0").core_profiles()
    result.deserialize(reply.data)
    assert all(result.time == core_profiles.time)

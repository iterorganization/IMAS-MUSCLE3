"""
MUSCLE3 actor overlaying one IDS on top of another.

Two IDSs of the same kind arrive per IDS name -- a base and an overlay --
and the merged result goes out: every node the overlay has a value for is
written over the base, and everything the overlay is silent about keeps the
base's value. The two IDSs must already be on the same time base.
"""

import logging
from typing import List

import numpy as np
from imas import IDSFactory
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance, InstanceFlags, Message
from ymmsl.v0_2 import Operator

from imas_muscle3.utils import ids_from_message

logger = logging.getLogger()

IDS_NAMES = IDSFactory().ids_names()


def base_port(ids_name: str) -> str:
    return f"{ids_name}_base"


def overlay_port(ids_name: str) -> str:
    return f"{ids_name}_overlay"


def out_port(ids_name: str) -> str:
    return f"{ids_name}_out"


def active_ids_names(instance: Instance) -> List[str]:
    """The IDS names this instance is wired for. An IDS needs all three of its
    ports connected.
    """
    active = []
    for ids_name in IDS_NAMES:
        ports = [
            base_port(ids_name),
            overlay_port(ids_name),
            out_port(ids_name),
        ]
        connected = [p for p in ports if instance.is_connected(p)]
        if not connected:
            continue
        if len(connected) != len(ports):
            missing = [p for p in ports if p not in connected]
            raise RuntimeError(
                f"merger: '{ids_name}' is only half wired: "
                f"{', '.join(connected)} connected, but "
                f"{', '.join(missing)} not. Connect all three, or none."
            )
        active.append(ids_name)

    if not active:
        raise RuntimeError(
            "merger: nothing is connected. Wire <ids_name>_base, "
            "<ids_name>_overlay and <ids_name>_out for at least one IDS."
        )
    logger.info("merger: merging %s", ", ".join(active))
    return active


def deserialize(data: bytes, ids_name: str, port: str) -> IDSToplevel:
    """The IDS carried by a message received on `port`."""
    if data is None:
        raise RuntimeError(
            f"merger: no data received on '{port}': nothing to merge"
        )
    return ids_from_message(ids_name, data)


def check_time(ids_name: str, base: IDSToplevel, overlay: IDSToplevel) -> None:
    """Refuse anything but two IDSs sharing one homogeneous time base."""

    for role, ids, port in (
        ("base", base, base_port(ids_name)),
        ("overlay", overlay, overlay_port(ids_name)),
    ):
        homogeneous = int(ids.ids_properties.homogeneous_time)
        if homogeneous != IDS_TIME_MODE_HOMOGENEOUS:
            raise RuntimeError(
                f"merger: the {role} '{ids_name}' on '{port}' is not in "
                "homogeneous time mode, so its root /time is not the time "
                "base of its data. This actor merges homogeneous IDSs only."
            )

    base_time = np.asarray(base.time, dtype=float)
    overlay_time = np.asarray(overlay.time, dtype=float)

    if base_time.size == 0 or overlay_time.size == 0:
        empty = "base" if base_time.size == 0 else "overlay"
        raise RuntimeError(
            f"merger: the {empty} '{ids_name}' has an empty root /time; "
            "there is no time base to merge on"
        )

    if base_time.shape != overlay_time.shape:
        raise RuntimeError(
            f"merger: '{ids_name}' base has {base_time.size} time step(s) "
            f"({base_time[0]:g}..{base_time[-1]:g}), overlay has "
            f"{overlay_time.size} "
            f"({overlay_time[0]:g}..{overlay_time[-1]:g}). Both sides must "
            "be on the same time base."
        )

    differing = np.flatnonzero(base_time != overlay_time)
    if differing.size:
        i = int(differing[0])
        raise RuntimeError(
            f"merger: '{ids_name}' the two time bases have "
            f"{base_time.size} step(s) each but differ from step {i} on: "
            f"base {float(base_time[i])!r} vs overlay "
            f"{float(overlay_time[i])!r} ({differing.size} step(s) differ "
            "in total)"
        )


def merge(base, overlay, ids_name: str, path: str = "") -> None:
    """Write every filled node of `overlay` into `base`, in place."""

    for child in overlay:
        if not child.has_value:
            continue
        name = child.metadata.name
        here = f"{path}/{name}" if path else name
        target = base[name]

        if isinstance(child, IDSStructArray):
            if len(child) > len(target):
                raise RuntimeError(
                    f"merger: '{ids_name}' the overlay's {here} has "
                    f"{len(child)} element(s) but the base's has "
                    f"{len(target)}. Merging would have to invent the "
                    "missing base elements, so it stops here instead: "
                    "either have the base supply them, or take this IDS "
                    "straight from its producer without merging."
                )
            for i, item in enumerate(child):
                merge(target[i], item, ids_name, f"{here}({i + 1})")
        elif isinstance(child, IDSStructure):
            merge(target, child, ids_name, here)
        else:
            base[name] = child.value


def main() -> None:
    """Create instance and merge each overlay IDS onto its base IDS."""
    logger.info("Starting merger actor")
    instance = Instance(
        {
            Operator.F_INIT: [base_port(n) for n in IDS_NAMES]
            + [overlay_port(n) for n in IDS_NAMES],
            Operator.O_F: [out_port(n) for n in IDS_NAMES],
        },
        flags=InstanceFlags.KEEPS_NO_STATE_FOR_NEXT_USE,
    )

    while instance.reuse_instance():
        ids_names = active_ids_names(instance)

        received = {
            ids_name: (
                instance.receive(base_port(ids_name)),
                instance.receive(overlay_port(ids_name)),
            )
            for ids_name in ids_names
        }

        for ids_name in ids_names:
            base_msg, overlay_msg = received[ids_name]

            if base_msg.timestamp != overlay_msg.timestamp:
                logger.warning(
                    "merger: '%s' base timestamp is at t=%s but overlay is at "
                    "t=%s; sending the base's timestamp",
                    ids_name,
                    base_msg.timestamp,
                    overlay_msg.timestamp,
                )

            base = deserialize(base_msg.data, ids_name, base_port(ids_name))
            overlay = deserialize(
                overlay_msg.data, ids_name, overlay_port(ids_name)
            )

            check_time(ids_name, base, overlay)
            merge(base, overlay, ids_name)
            logger.info(
                "merger: merged '%s' on %d time step(s)",
                ids_name,
                np.asarray(base.time).size,
            )

            instance.send(
                out_port(ids_name),
                Message(
                    base_msg.timestamp,
                    base_msg.next_timestamp,
                    base.serialize(),
                ),
            )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

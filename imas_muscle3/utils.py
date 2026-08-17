import re
from typing import List, Optional, TypeVar, cast
from urllib.parse import urlparse, urlunparse

from imas import IDSFactory
from imas.ids_defs import CLOSEST_INTERP, LINEAR_INTERP, PREVIOUS_INTERP
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance

# ymmsl 0.15+ split into versioned subpackages; import the v0.2 API explicitly.
from ymmsl.v0_2 import Operator, SettingValue

TSetting = TypeVar("TSetting", bound=SettingValue)


def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: Optional[TSetting] = None,
) -> Optional[TSetting]:
    """Helper function to get optional settings from instance.

    libmuscle's Instance.get_setting(default=...) cannot distinguish
    "no default" from "default is None", so it re-raises KeyError even
    when default=None is passed explicitly. This wraps it correctly.
    """
    setting: Optional[TSetting]
    try:
        setting = cast(TSetting, instance.get_setting(setting_name))
    except KeyError:
        setting = default
    return setting


def get_port_list(instance: Instance, operator: Operator) -> List[str]:
    """Sorted list of ids_names by which ones are actually connected for
    given instance"""
    total_port_list = instance.list_ports().get(operator, [])
    port_list = sorted(
        port for port in total_port_list if instance.is_connected(port)
    )
    return port_list


def ids_name_from_port(port_name: str) -> str:
    """The IDS name a port carries: its name, optional ``_in`` stripped."""
    ids_name = port_name[:-3] if port_name.endswith("_in") else port_name
    if ids_name not in IDSFactory().ids_names():
        raise ValueError(
            f"Port '{port_name}' does not map to a known IDS name "
            f"(resolved to '{ids_name}'). Name the port after the IDS it "
            f"carries, optionally with an '_in' suffix."
        )
    return ids_name


def fix_interpolation_method(instance: Instance) -> int:
    setting = instance.get_setting("interpolation_method", default="closest")
    if setting == "closest":
        interp = CLOSEST_INTERP
    elif setting == "previous":
        interp = PREVIOUS_INTERP
    elif setting == "linear":
        interp = LINEAR_INTERP
    else:
        interp = CLOSEST_INTERP
    return interp


def ids_from_message(ids_name: str, data: bytes) -> IDSToplevel:
    """Deserialize a received message's payload into a fresh IDS."""
    ids = IDSFactory().new(ids_name)
    ids.deserialize(data)
    return ids


def increment_suffix(uri: str) -> str:
    # parse uri
    parsed = urlparse(uri)
    query_dict = {}
    for option in re.split("[&;?]", parsed.query):
        name, _, value = option.partition("=")
        query_dict[name] = value

    # increment path
    path = query_dict["path"]
    if "_" in path and path.rsplit("_", 1)[-1].isdigit():
        base, num = path.rsplit("_", 1)
        query_dict["path"] = f"{base}_{int(num) + 1}"
    else:
        query_dict["path"] = f"{path}_1"

    # rebuild uri
    new_query = "&".join(f"{k}={v}" for k, v in query_dict.items())
    parsed = parsed._replace(query=new_query)
    return urlunparse(parsed)

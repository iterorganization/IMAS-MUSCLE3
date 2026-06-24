import re
from typing import List, Optional, TypeVar, cast, overload
from urllib.parse import urlparse, urlunparse

from libmuscle import Instance

# ymmsl 0.15+ split into versioned subpackages; import the v0.2 API explicitly.
from ymmsl.v0_2 import Operator, SettingValue

TSetting = TypeVar("TSetting", bound=SettingValue)


@overload
def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: None = None,
) -> Optional[TSetting]: ...


@overload
def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: TSetting,
) -> TSetting: ...


# it may be a nice proposal for the m3 api
def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: Optional[TSetting] = None,
) -> Optional[TSetting]:
    """Helper function to get optional settings from instance"""
    setting: Optional[TSetting]
    try:
        setting = cast(TSetting, instance.get_setting(setting_name))
    except KeyError:
        setting = default
    return setting


def get_port_list(instance: Instance, operator: Operator) -> List[str]:
    """Filter list of ids_names by which ones are actually connected for
    given instance"""
    total_port_list = instance.list_ports().get(operator, [])
    port_list = [
        port for port in total_port_list if instance.is_connected(port)
    ]
    return port_list


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

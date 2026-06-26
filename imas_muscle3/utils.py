from typing import List, Optional, TypeVar, cast

from libmuscle import Instance

# ymmsl 0.15+ split into versioned subpackages; import the v0.2 API explicitly.
from ymmsl.v0_2 import Operator, SettingValue

TSetting = TypeVar("TSetting", bound=SettingValue)


def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: Optional[TSetting] = None,
) -> Optional[TSetting]:
    """Get an optional setting, returning ``default`` when unset.

    libmuscle's ``Instance.get_setting(default=...)`` re-raises ``KeyError``
    when the default is ``None``, so it cannot express "optional, may be None".
    This helper covers that case.
    """
    try:
        return cast(TSetting, instance.get_setting(setting_name))
    except KeyError:
        return default


def get_port_list(instance: Instance, operator: Operator) -> List[str]:
    """Filter list of ids_names by which ones are actually connected for
    given instance"""
    total_port_list = instance.list_ports().get(operator, [])
    port_list = [
        port for port in total_port_list if instance.is_connected(port)
    ]
    return port_list

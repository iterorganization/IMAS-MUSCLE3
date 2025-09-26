from typing import List, Optional, TypeVar, cast

from libmuscle import Instance
from ymmsl import Operator, SettingValue

TSetting = TypeVar("TSetting", bound=SettingValue)


# it may be a nice proposal for the m3 api
def get_setting_optional(
    instance: Instance,
    setting_name: str,
    default: Optional[SettingValue] = None,
) -> Optional[SettingValue]:
    """Helper function to get optional settings from instance"""
    setting: Optional[SettingValue]
    try:
        setting = instance.get_setting(setting_name)
    except KeyError:
        setting = default
    return setting


def get_setting_default(
    instance: "Instance",
    setting_name: str,
    default: TSetting,
) -> TSetting:
    """Helper function to get settings from instance"""
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

from typing import List, Optional, TypeVar, cast

from libmuscle import Instance

# ymmsl 0.15+ split into versioned subpackages; import the v0.2 API explicitly.
from ymmsl.v0_2 import Operator, SettingValue

TSetting = TypeVar("TSetting", bound=SettingValue)


def get_port_list(instance: Instance, operator: Operator) -> List[str]:
    """Filter list of ids_names by which ones are actually connected for
    given instance"""
    total_port_list = instance.list_ports().get(operator, [])
    port_list = [
        port for port in total_port_list if instance.is_connected(port)
    ]
    return port_list

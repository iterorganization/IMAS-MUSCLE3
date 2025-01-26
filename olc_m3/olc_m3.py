r"""
MUSCLE3 actor performing limit checking with IDS-validator.

On message arrival on an F_INIT or S port validation is launched for that IDS.
Multi-IDS validation is only performed if messages with all those IDSes arrive
with the same timestamp.
"""

import sys, time
from ymmsl import Operator
from libmuscle import Instance, Message
import logging

import imaspy
from imaspy import DBEntry, IDSFactory

from ids_validator.validate_options import ValidateOptions
from ids_validator.validate import validate


logger = logging.getLogger()
IMAS_URI = "imas:memory?path=/"

def main():
    """Create instance and enter submodel execution loop"""
    logger.info("Starting OLC Actor")
    instance = Instance(
        {
            Operator.F_INIT: [
                f"{ids_name}_in" for ids_name in IDSFactory().ids_names()
            ],
        }
    )

    # enter re-use loop
    while instance.reuse_instance():
        port_list_in = get_port_list(instance, Operator.F_INIT)
        # wait for messages on all of the ports (one after the other)
        # Note: this does not handle uneven message counts on different ports.
        # nor does it handle occurrances
        # but it must be this way now, since there is no `select` for muscle3 ports
        ids_data = {}
        for port_name in port_list:
            ids_name = port_name.replace("_in", "")
            msg_in = instance.receive(port_name)
            t_cur = msg_in.timestamp
            ids_data[ids_name] = getattr(IDSFactory(), ids_name)()
            ids_data[ids_name].deserialize(msg_in.data)

        # we have now received one message on each of the ports, and can launch a validation action
        with DBEntry(IMAS_URI, 'w') as db:
            # write all IDSes to the memory entry, since ids_validator prefers to load stuff itself.
            # some performance improvement could be made there by making a second entrypoint
            # that does not load by imas_uri but accepts a collection of toplevels.
            for ids in ids_data:
                db.put(ids)

            validate_options = ValidateOptions(
              rulesets = get_setting_optional(instance, f"rulesets", default=['PDS-OLC']),
              extra_rule_dirs = get_setting_optional(instance, f"extra_rule_dirs", default=[]),
              apply_generic = get_setting_optional(instance, f"apply_generic", default=True),
              use_pdb = get_setting_optional(instance, f"use_pdb", default=False),
              rule_filter = RuleFilter(ids = ids_data.keys()), # this one may not be needed
            )

            result = validate(IMAS_URI, validate_options)
            validation_passed = all(
                [r.success for r in r.results]
            )
            if validation_passed:
                logger.info("Check passed")
            else:
                today = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                # not sure whether we need the summary report generator or the detailed one
                summary_generator = SummaryReportGenerator(result, today)
                summary_generator.save_html(f'{t_cur}_report.html')

                if get_setting_optional(instance, f"halt_on_error", default=False):
                    logger.critical("Check failed!")
                    sys.exit(1)
                else:
                    # this message can be much more verbose. Should include:
                    # - IDSes that have failed
                    # - rules that have been violated (unless there are many?)
                    logger.warning("Check failed!")

# the below helper methods also occur in https://git.iter.org/projects/SCEN/repos/pds/browse/pds/utils/data_sink_source.py

# it may be a nice proposal for the m3 api
def get_setting_optional(
    instance: Instance, setting_name: str, default: Optional[SettingValue] = None
) -> Optional[SettingValue]:
    """Helper function to get optional settings from instance"""
    setting: Optional[SettingValue]
    try:
        setting = instance.get_setting(setting_name)
    except KeyError:
        setting = default
    return setting

def get_port_list(instance: Instance, operator: Operator) -> List[str]:
    """Filter list of ids_names by which ones are actually connected for
    given instance"""
    total_port_list = instance.list_ports().get(operator, [])
    port_list = [port for port in total_port_list if instance.is_connected(port)]
    return port_list


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

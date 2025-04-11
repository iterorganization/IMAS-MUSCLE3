r"""
MUSCLE3 actor performing Operational Limit Checking with IDS-validator.

On message arrival on an F_INIT or S port validation is launched for that IDS.
Multi-IDS validation is only performed if messages with all those IDSes arrive
with the same timestamp.
"""

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from ids_validator.report.summaryReportGenerator import SummaryReportGenerator
from ids_validator.validate.validate import validate
from ids_validator.validate_options import RuleFilter, ValidateOptions
from imas import DBEntry, IDSFactory
from libmuscle import Instance
from ymmsl import Operator

from imas_m3.utils import get_port_list, get_setting_optional

logger = logging.getLogger()


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
        for port_name in port_list_in:
            ids_name = port_name.replace("_in", "")
            msg_in = instance.receive(port_name)
            t_cur = msg_in.timestamp
            ids_data[ids_name] = getattr(IDSFactory(), ids_name)()
            ids_data[ids_name].deserialize(msg_in.data)

        # we have now received one message on each of the ports, and can launch a
        # validation action
        with tempfile.TemporaryDirectory() as tmpdir:
            IMAS_URI = f"imas:hdf5?path={tmpdir}"
            with DBEntry(IMAS_URI, "w") as db:
                # write all IDSes to the temporary HDF5 entry, since ids_validator prefers to load
                # stuff itself. Some performance improvement could be made there by making
                # a second entrypoint that does not load by imas_uri but accepts a
                # collection of toplevels.
                for ids in ids_data.values():
                    db.put(ids)

            validate_options = ValidateOptions(
                rulesets=get_setting_optional(
                    instance, "rulesets", default="PDS-OLC"
                ).split(";"),
                extra_rule_dirs=[
                    Path(x)
                    for x in get_setting_optional(
                        instance, "extra_rule_dirs", default=""
                    ).split(";")
                ],
                apply_generic=get_setting_optional(
                    instance, "apply_generic", default=True
                ),
            )

            result = validate(IMAS_URI, validate_options)
            validation_passed = all([r.success for r in result.results])
            if validation_passed:
                logger.info("Check passed")
            else:
                today = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                # not sure whether we need the summary report generator or the detailed
                summary_generator = SummaryReportGenerator([result], today)
                summary_generator.save_html(f"{t_cur}_report.html")

                if get_setting_optional(instance, "halt_on_error", default=False):
                    logger.critical("Check failed!")
                    sys.exit(1)
                else:
                    # this message can be much more verbose. Should include:
                    # - IDSes that have failed
                    # - rules that have been violated (unless there are many?)
                    logger.warning("Check failed!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

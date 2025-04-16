.. _`actor_olc`:

OLC actor
=================

Actor for Operational Limit Checking of IMAS data in a simulation through the IDS-validator tool.
Useful for testing whether the simulation is still physical or to check whether a given workflow is viable for experiments.

Available Settings
------------------

* Optional
  - ***halt_on_error***: (bool) Whether or not the simulation should be forcibly stopped when a validation test fails. Defaults to False.
  - ***extra_rule_dirs***: (str) The rule directories in which to look for IDS-validator rulesets. If inserting multiple, split them with a ';'. Defaults to ''.
  - ***rulesets***: (str) The names of rulesets to run in the found rule directories. If inserting multiple, split them with a ';'. Defaults to 'PDS-OLC'.
  - ***apply-generic***: (bool) Whether or not to apply the generic bundled validation tests. Defaults to True.

Available Ports
---------------
All IDS's are available for the OLC actor. They will be active if connected in the ymmsl file and will be skipped otherwise.

* Optional
- ***<ids_name>_in (F_INIT)***: Any incoming IDS's on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in.

General
-------
The OLC actor is not bound to a specific DD version.

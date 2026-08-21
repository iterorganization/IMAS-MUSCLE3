.. _`actor_source`:

Source actor
============

Actor for loading generic IMAS data in a simulation from disk. Useful for providing starting conditions
for simulations when creating simulation workflows, as well as debugging and testing purposes.

Use an IDS as a data source. Loops over the timestamps and sends them out one by one.

.. code-block:: bash

  implementations:
    source_component:
      executable: python
      args: -u -m imas_muscle3.actors.source_component

Available Settings
------------------

* Mandatory

  - **source_uri**: (string) IMAS URI from which to load data.

* Optional

  - **dd_version**: (string) IMAS Data Dictionary version number to which data will be converted. Defaults to original dd_version of the data.
  - **<ids_name>_occ**: (int) Occurence number to load from for a given ids_name. Replace <ids_name> with the required ids i.e. equilibrium_occ. Defaults to 0.
  - **t_min**: (float) Minimum time value for loading timeslices. Defaults to None.
  - **t_max**: (float) Maximum time value for loading timeslices. Defaults to None.
  - **dt**: (float) Fixed time step. When set, an evenly-spaced time array is generated
    over the (t_min/t_max-bounded) source range instead of using the source IDS's native timestamps. Defaults to None.
  - **interpolation_method**: (string) Which `IMAS interpolation method <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.get_sample.interpolation_method>`_ to use for source.
    Can choose from "closest", "previous", "linear". Defaults to "closest".
  - **iterative**: (bool) True loops over all timeslices, False sends them all at once. Defaults to True.

Available Ports
---------------

All IDS's are available for the source actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
The source actor uses only the O_I port.

* Optional

  - **<ids_name>_out (O_I)**: Any outgoing IDS's on the O_I port. Replace <ids_name> with the required ids i.e. equilibrium_out.

General
-------
The source actor is not bound to a specific DD version.

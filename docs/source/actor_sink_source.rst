.. _`actor_sink_source`:

Sink/source actor
=================

Actors for loading and saving generic IMAS data in a simulation. Useful for debugging and testing purposes
when creating simulation workflows, as well as providing starting conditions and saving results for simulations.

Available Operational Modes
---------------------------

- **Source actor**: Use an IDS as a data source. Loop over the timestamps and send them out one by one.

.. code-block:: bash

  implementations:
    source_component:
      executable: python
      args: -u -m imas_muscle3.actors.source_component

- **Sink actor**: Use an IDS as a data sink. Saves all incoming data to an IDS.

.. code-block:: bash

  implementations:
    sink_component:
      executable: python
      args: -u -m imas_muscle3.actors.sink_component

- **Combined actor**: Receives data, optionally saves it and sends out preexisting IDS data for the timestamp closest to the incoming timestamp.

.. code-block:: bash

  implementations:
    sink_source_component:
      executable: python
      args: -u -m imas_muscle3.actors.sink_source_component

Available Settings
------------------

* Mandatory

  - **sink_uri**: (string) IMAS URI in which to save incoming data. (Only sink and sink_source)
  - **source_uri**: (string) IMAS URI from which to load data. (Only source and sink_source)

* Optional

  - **dd_version**: (string) IMAS Data Dictionary version number to which data will be converted. Defaults to original dd_version of the data.
  - **<ids_name>_occ**: (int) Occurence number to load from or save to for a given ids_name. Replace <ids_name> with the required ids i.e. equilibrium_occ. Defaults to 0.
  - **t_min**: (float) Minimum time value for loading timeslices. Defaults to None.
  - **t_max**: (float) Maximum time value for loading timeslices. Defaults to None.
  - **interpolation_method**: (int) Which `IMAS interpolation method <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.get_sample.interpolation_method>`_ to use for source. Defaults to CLOSEST_INTERP.
  - **sink_mode**: (string) Mode argument for `DBEntry <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.__init__.mode>`_. 'w' means you always overwrite your full data entry. 'x' means you are not allowed to overwrite old data. Defaults to 'x'.
  - **iterative**: (bool) True loops over all timeslices, False sends them all at once. Defaults to True.

Available Ports
---------------

All IDS's are available for the sink/source actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
The source actor uses only the O_I port. The sink actor uses only the F_INIT port. The combined actor uses the F_INIT and O_F ports.

* Optional

  - **<ids_name>_in (F_INIT)**: Any incoming IDS's on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in.
  - **<ids_name>_out (O_F, O_I)**: Any outgoing IDS's on the O_F and O_I ports. Replace <ids_name> with the required ids i.e. equilibrium_out.

General
-------
The sink/source actor is not bound to a specific DD version.

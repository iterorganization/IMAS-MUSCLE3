.. _`actor_sink_source`:

Sink/source actor
=================

Combined actor for loading and saving generic IMAS data in a simulation to disk. Useful for providing starting
conditions and saving results for simulations when creating simulation workflows, as well as debugging and
testing purposes.

Receives data, optionally saves it and sends out preexisting IDS data for the timestamp closest to the incoming
timestamp.

.. code-block:: bash

  implementations:
    sink_source_component:
      executable: python
      args: -u -m imas_muscle3.actors.sink_source_component

Available Settings
------------------

* Mandatory

  - **source_uri**: (string) IMAS URI from which to load data.

* Optional

  - **sink_uri**: (string) IMAS URI in which to save incoming data. If unset, this instance only acts as a source.
  - **dd_version**: (string) IMAS Data Dictionary version number to which data will be converted. Defaults to original dd_version of the data.
  - **<ids_name>_occ**: (int) Occurence number to load from or save to for a given ids_name. Replace <ids_name> with the required ids i.e. equilibrium_occ. Defaults to 0.
  - **interpolation_method**: (string) Which `IMAS interpolation method <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.get_sample.interpolation_method>`_ to use for source.
    Can choose from "closest", "previous", "linear". Defaults to "closest".
  - **sink_mode**: (string) Mode argument for `DBEntry <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.__init__.mode>`_. 'w' means you always overwrite your full data entry. 'x' means you are not allowed to overwrite old data. Defaults to 'x'.
  - **avoid_name_collision**: (bool) True means that if a given DBEntry uri is already found and you are on sink_mode 'x',
    it creates a new numbered path. False means it raises an error. Defaults to True.
    (i.e. "imas:hdf5?path=my/path" is converted to "imas:hdf5?path=my/path_1")

Available Ports
---------------

All IDS's are available for the combined actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
The combined actor uses the F_INIT and O_F ports.

* Optional

  - **<ids_name>_in (F_INIT)**: Any incoming IDS's on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in.
  - **<ids_name>_out (O_F)**: Any outgoing IDS's on the O_F port. Replace <ids_name> with the required ids i.e. equilibrium_out.

General
-------
The sink/source actor is not bound to a specific DD version.

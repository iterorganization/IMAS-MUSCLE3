.. _`actor_sink`:

Sink actor
==========

Actor for saving generic IMAS data in a simulation to disk. Useful for saving results
for simulations when creating simulation workflows, as well as debugging and testing purposes.

Use an IDS as a data sink. Saves all incoming data to an IDS.

.. code-block:: bash

  implementations:
    sink_component:
      executable: python
      args: -u -m imas_muscle3.actors.sink_component

Available Settings
------------------

* Mandatory

  - **sink_uri**: (string) IMAS URI in which to save incoming data.

* Optional

  - **dd_version**: (string) IMAS Data Dictionary version number to which data will be converted. Defaults to original dd_version of the data.
  - **<ids_name>_occ**: (int) Occurence number to save to for a given ids_name. Replace <ids_name> with the required ids i.e. equilibrium_occ. Defaults to 0.
  - **sink_mode**: (string) Mode argument for `DBEntry <https://imas-python.readthedocs.io/en/stable/generated/imas.db_entry.DBEntry.html#imas.db_entry.DBEntry.__init__.mode>`_. 'w' means you always overwrite your full data entry. 'x' means you are not allowed to overwrite old data. Defaults to 'x'.
  - **avoid_name_collision**: (bool) True means that if a given DBEntry uri is already found and you are on sink_mode 'x',
    it creates a new numbered path. False means it raises an error. Defaults to True.
    (i.e. "imas:hdf5?path=my/path" is converted to "imas:hdf5?path=my/path_1")

Available Ports
---------------

All IDS's are available for the sink actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
The sink actor uses only the F_INIT port.

* Optional

  - **<ids_name>_in (F_INIT)**: Any incoming IDS's on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in.

General
-------
The sink actor is not bound to a specific DD version.

.. _`actor_tap`:

Tap recorder actor
==================

A terminal (sink-only) actor that taps onto an arbitrary number of independent
*timelines* at once and records every message it receives to disk, one DBEntry
per message. It captures the live traffic of a running workflow for later
inspection without disturbing the coupling. For a compact, plot-ready recording
instead, see the :ref:`distill recorder <actor_distill>`.

.. code-block:: bash

  implementations:
    tap_component:
      executable: python
      args: -u -m imas_muscle3.actors.tap_component

.. include:: _recorder_model.rst

Storage layout
--------------

Each received message is written to its own DBEntry at::

  imas:hdf5?path=<store_path>/<port_name>/<seq>

where ``<seq>`` is a zero-padded, per-timeline sequence number. ``store_path``
defaults to the instance's run folder (its working directory in the MUSCLE3
run), so by default a tap records into its own component directory. Recordings
read back with the usual IMAS-Python tooling::

  imas.DBEntry("imas:hdf5?path=<store_path>/equilibrium_in/00000000", "r").get("equilibrium")

The full IMAS URI of every recorded message is logged at ``INFO`` level so it can
be pasted straight into ``imas.DBEntry(...)``. There is no storage cap: the tap
records everything it receives.

Available Settings
------------------

* Optional

  - **store_path**: (string) Directory root under which per-message DBEntries
    are written. Defaults to the instance's run folder (its working directory).

Available Ports
---------------

* Optional

  - **<ids_name>[_in] (S)**: Any incoming IDS on the S operator. The port name
    is the IDS name, optionally suffixed with ``_in``.

General
-------
The tap recorder is not bound to a specific DD version.

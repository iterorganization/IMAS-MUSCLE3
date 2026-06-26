.. _`actor_recorder`:

Recorder actors
===============

Two terminal (sink-only) actors that tap the live traffic of a running workflow
and write it to disk, without disturbing the coupling:

- **tap** (``imas_muscle3.actors.tap_component``) stores each received IDS
  verbatim, one DBEntry per message — a faithful, re-openable copy.
- **distill** (``imas_muscle3.actors.distill_component``) reduces each IDS to
  compact scalars, profiles and maps in a Zarr store — a small, self-describing
  dataset a viewer can plot live or open afterwards.

.. code-block:: bash

  implementations:
    tap_component:
      executable: python
      args: -u -m imas_muscle3.actors.tap_component
    distill_component:
      executable: python
      args: -u -m imas_muscle3.actors.distill_component

.. include:: _recorder_model.rst

Output
------

**tap** writes each message to its own DBEntry, re-openable with IMAS-Python::

  imas:hdf5?path=<store_path>/<port_name>/<seq>      # zero-padded <seq>

**distill** records each outer-loop iteration as its own *occurrence*, so
iterations sit side by side::

  <store_path>/<port_name>/<NNNN>.zarr               # open_zarr(..., group=<ids>)

``store_path`` defaults to the instance's run folder.

Settings
--------

* Optional

  - **store_path**: (string) Root for the output. Defaults to the run folder.
  - **auto** (distill): (bool) Record every time-dependent 0D/1D/2D ``FLT``
    quantity (GGD/grid excluded). Defaults to ``true``.
  - **config** (distill): (string) Path to a Python file defining
    ``extract(ids) -> dict[str, xarray.Dataset]`` for derived/geometric
    quantities, recorded alongside (or, with ``auto: false``, instead of) the
    auto-discovered ones.

Ports
-----

Any connected ``S`` port; its name is the IDS it carries, optionally ``_in``-
suffixed. Not bound to a specific DD version.

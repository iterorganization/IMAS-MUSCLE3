.. _`actor_distill`:

Distill recorder actor
======================

A terminal (sink-only) actor that taps the same independent *timelines* as the
:ref:`tap recorder <actor_tap>`, but instead of storing each raw IDS it
*distills* every message into compact scalars / profiles / maps and appends them
along ``time`` to a Zarr store. The result is a small, self-describing,
append-only dataset that a viewer can plot live as the run writes it, or open
afterwards.

.. code-block:: bash

  implementations:
    distill_component:
      executable: python
      args: -u -m imas_muscle3.actors.distill_component

.. include:: _recorder_model.rst

Distillation
-----------

Each received IDS is reduced to the quantities worth plotting and tensorized with
imas-python's :func:`imas.util.to_xarray` (so the datasets follow imas-python's
netCDF conventions). Two sources are combined:

- **auto-discovery** (``auto``, default ``true``): every time-dependent 0D/1D/2D
  ``FLT`` quantity in the IDS (GGD/grid subtrees are skipped, as they explode
  into tens of thousands of nodes).
- a **config callable** (``config``): a Python file defining ``extract(ids) ->
  dict[str, xarray.Dataset]`` for derived/geometric quantities (separatrix,
  contours, ...) that auto-discovery cannot express.

Storage layout
--------------

Output is split into one *occurrence* per outer-loop iteration::

  <store_path>/<port_name>/<NNNN>.zarr

A new occurrence begins whenever the timeline restarts -- a message with no
``next_timestamp``, or simulation time stepping backwards onto the same grid --
so each Picard/outer-loop iteration lands in its own store for side-by-side
comparison, derived purely from the message stream with no extra wiring.
``store_path`` defaults to the instance's run folder. Reopen with::

  xarray.open_zarr("<store_path>/equilibrium_in/0000.zarr", group="equilibrium")

Available Settings
------------------

* Optional

  - **store_path**: (string) Directory root for the per-port Zarr stores.
    Defaults to the instance's run folder (its working directory).
  - **auto**: (bool) Auto-discover and record every time-dependent 0D/1D/2D
    ``FLT`` quantity. Defaults to ``true``.
  - **config**: (string) Path to a Python file defining ``extract(ids) ->
    dict[str, xarray.Dataset]`` for derived quantities, recorded in addition to
    (or, with ``auto: false``, instead of) the auto-discovered ones.

Available Ports
---------------

* Optional

  - **<ids_name>[_in] (S)**: Any incoming IDS on the S operator. The port name
    is the IDS name, optionally suffixed with ``_in``.

General
-------
The distill recorder is not bound to a specific DD version.

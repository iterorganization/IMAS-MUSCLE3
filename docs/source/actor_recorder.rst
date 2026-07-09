.. _`actor_recorder`:

Recorder actor
==============

A terminal (sink-only) actor that taps the live traffic of a running workflow
and distills it to disk, without disturbing the coupling: wire it as an extra
receiver on existing conduits (a fan-out entry in the conduit list), so it
adds no conduit the producers wait on. It uses MUSCLE3 dynamic ports: every
connected ``S`` port is an independent timeline, its name the IDS it carries
(an optional ``_in`` suffix is stripped).

Each received IDS is reduced to plot-ready ``xarray`` datasets by the
``config`` file and appended to a Zarr store that can be read back — also
mid-run, for live views — with ``xarray.open_zarr(store, group=<name>)``.
A driven sender re-running each outer-loop iteration is recorded across every
iteration, one store per occurrence: ``<store_path>/<port>/<NNNN>.zarr``.

.. code-block:: yaml

    components:
      rec:
        implementation: recorder
        ports:
          s: [equilibrium_in, pf_active_in]
    settings:
      rec.config: /path/to/plot_file.py
    implementations:
      recorder:
        executable: python
        args: -u -m imas_muscle3.actors.recorder_component

Config file
-----------

The ``config`` setting names a Python file defining either:

* ``extract(ids) -> dict[str, xarray.Dataset]`` — every dataset carries a
  ``time`` dimension (one instant or a whole trace) to append along; or
* a ``State`` class (a
  :class:`~imas_muscle3.visualization.base_state.BaseState` subclass, i.e. a
  :ref:`visualization actor <actor_visualization>` plot file) — each message
  is fed to a fresh instance and its accumulated ``data`` datasets are
  recorded.

The ``State`` form is how the muscle3-dashboard renders recorder data: the
same plot file defines what is stored (``State``) and how it is plotted
(``Plotter``), so the stored quantities and the plots that read them stay in
lockstep. The dashboard finds the file through the run's ``<rec>.config``
setting (each store's root attributes also carry it as ``distill_profile``)
and shows one tab per recorder, live while the run appends or after it
finished.

Settings
--------

* **config** (required): path to the extraction file described above.
* **store_path** (optional): root for the output. Defaults to the instance's
  run folder, where the dashboard looks for recorder stores.

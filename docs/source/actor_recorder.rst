.. _`actor_recorder`:

Recorder actor
==============

A terminal (sink-only) actor that taps the live traffic of a running workflow
without disturbing the coupling: wire it as an extra receiver on existing
conduits. Every connected ``S`` port is an independent timeline, its name the
IDS it carries (an optional ``_in`` suffix is stripped). Each received IDS is
reduced to plot-ready ``xarray`` datasets by the ``config`` file and appended
to a Zarr store that can be read back — also mid-run, for live views. Each
outer-loop iteration gets its own store:
``<store_path>/<port>/<iteration_number>.zarr``.

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

Available Settings
------------------

* Mandatory

  - **config** (str): path to the extraction file described above.

* Optional

  - **store_path** (str): root for the output. Defaults to the instance's
    run folder, where the dashboard looks for recorder stores.

Available Ports
---------------

  - **<ids_name>_in (S)**: Any incoming IDS's on the S port. Replace <ids_name> with the required ids i.e. equilibrium_in.

Config file
-----------

The ``config`` setting names a Python file defining either:

* ``extract(ids) -> dict[str, xarray.Dataset]`` — every dataset carries a
  ``time`` dimension (one instant or a whole trace) to append along; or
* a ``State`` class (a :ref:`visualization actor <actor_visualization>` plot
  file) — each message is fed to a fresh instance and its accumulated ``data``
  datasets are recorded.

The ``State`` form lets one plot file define both what is stored and how the
muscle3-dashboard plots it. On startup the recorder copies the config file
next to the data (``<store_path>/<config name>``) and stamps each store's
``distill_profile`` attribute with the copy's path; the dashboard prefers
this snapshot over the run's ``<rec>.config`` setting, so a recorded run
keeps plotting with the exact code that produced it even after the original
file is edited.

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
  - **automatic_extract** (bool): fill in extraction via
    ``BaseState.automatic_extract`` when ``config``'s ``State`` doesn't
    implement its own ``extract`` (see `Automatic extraction`_ below).
  - **automatic_extract_fields** (str): whitespace-separated full paths
    (``<ids_name>/<path>``, or the config's own keys) to restrict the
    recording to. Works with any config, not just an ``automatic_extract``
    one; default: keep everything the config extracts.

Available Ports
---------------

  - **<ids_name>_in (S)**: Any incoming IDS's on the S port. Replace <ids_name> with the required ids i.e. equilibrium_in.

Config file
-----------

The ``config`` setting names a Python file defining the extraction logic:
what gets pulled out of each received IDS and written to a store.
There are two ways to write it.

**Plain extract function**

Define a module-level function::

    def extract(ids) -> dict[str, xarray.Dataset]:
        ...

Every returned dataset must carry a ``time`` dimension (one instant, or a
whole trace) to append along. Use this form when the file only needs to
record data, with no plotting attached.

**State class (shared with the visualization actor)**

Define a ``State(BaseState)`` class implementing ``extract(self, ids)`` —
the same class a :ref:`visualization actor <actor_visualization>` plot file
uses. The recorder builds a *fresh* ``State`` instance for every received
message, calls its ``extract(self, ids)``, and records whatever ended up in
``self.data`` from that one call.

Because the instance is fresh each time, a ``State`` that accumulates
across calls (as the live visualization actor's does, concatenating onto
``self.data`` from one message to the next) only accumulates *within* one
``extract`` call. Both message granularities still end up fully recorded,
just via a different layer:

* One time slice per message: each call records a single instant; the
  recorder's own Zarr store does the accumulating, appending each new
  instant to the same occurrence.
* A whole trace per message (e.g. one message per Picard iteration,
  looping over ``ids.time_slice`` inside ``extract``): each call already
  records the full trace it was given; the Zarr store then appends
  whole-trace batches instead of single instants.

A ``State``-only file works for recording, but there's nothing to plot it
with. Add a ``Plotter(BasePlotter)`` class to the same file (see
:ref:`actor_visualization` for the full ``State``/``Plotter`` contract) and
it becomes an ordinary visualization actor plot file too — the recorder
still only reads the ``State`` half and ignores ``Plotter``, but the exact
same file can then be pointed at by a visualization actor, or read by a
dashboard, to plot precisely what was recorded, with no separate extraction
logic to keep in sync.

Automatic extraction
---------------------

A ``config`` file still always has to exist -- in particular, a ``Plotter``
is not something the recorder or a dashboard can invent for you, so one
always has to be hand-written. What can be automatic is the ``State`` behind
it: if ``config``'s ``State`` does not implement its own ``extract`` (for
example, one written only to hold data for a ``Plotter``, with no extraction
logic of its own) and ``automatic_extract: true`` is set, extraction falls
back to ``BaseState.automatic_extract`` -- the same discovery-and-extract
logic the live visualization actor's own automatic mode uses, with no
IDS-specific code required:

.. code-block:: python

    from imas_muscle3.visualization.base_state import BaseState
    from imas_muscle3.visualization.base_plotter import BasePlotter


    class State(BaseState):
        pass  # extraction is filled in automatically


    class Plotter(BasePlotter):
        def get_dashboard(self):
            ...

.. code-block:: yaml

    settings:
      rec.config: /path/to/plot_file.py
      rec.automatic_extract: true
      rec.automatic_extract_fields: equilibrium.time_slice[0].global_quantities.ip

This fallback is opt-in -- without ``automatic_extract: true``, a ``State``
that doesn't implement ``extract`` fails loudly, since it might simply be a
mistake rather than an intentional automatic-mode config.

Since automatic extraction has no way to know in advance which quantities you
care about, it discovers and records *everything*
``BaseState.automatic_extract`` finds unless ``automatic_extract_fields``
restricts it to a handful of named ones. Note the dots, not slashes:
``automatic_extract``'s discovered full paths (e.g.
``equilibrium/time_slice[0]/global_quantities/ip``) are flattened to ``.``
before recording, since Zarr rejects ``/`` in a variable name --
``automatic_extract_fields`` (and the on-disk group name) must match that
same dotted form.

**Config snapshotting**

On startup the recorder copies the config file next to the data
(``<store_path>/<config name>``) and stamps each store's
``distill_profile`` attribute with the copy's path. A dashboard reading
these stores prefers this snapshot over the run's ``<rec>.config`` setting,
so a recorded run keeps plotting with the exact code that produced it even
after the original file is edited.

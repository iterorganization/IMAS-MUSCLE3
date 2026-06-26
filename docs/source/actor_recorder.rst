.. _`actor_recorder`:

Recorder actor
==============

A terminal (sink-only) actor that taps the live traffic of a running workflow
and writes it to disk without disturbing the coupling. It uses MUSCLE3 dynamic
ports: every connected ``S`` port is an independent timeline, its name the IDS
it carries (an optional ``_in`` suffix is stripped). Each timeline is written as
its messages arrive, in full until that port's stream closes; a driven sender
re-running each outer-loop iteration is recorded across every iteration, one
store per occurrence ``NNNN``.

.. code-block:: yaml

    implementations:
      recorder:
        executable: python
        args: -u -m imas_muscle3.actors.recorder_component

Formats
-------

The ``format`` setting picks what is written:

* ``imas`` (default): each IDS verbatim as an IMAS DBEntry, re-openable with
  ``imas.DBEntry("imas:hdf5?path=<...>/0000", "r")``.
* ``raw``: each MUSCLE3 message (frame + undecoded payload) as a msgpack stream
  of ``{t, next_t, data}`` records, read via ``msgpack.Unpacker``; the cheapest,
  most faithful capture.
* ``distill``: each IDS reduced to compact arrays in a Zarr store, opened with
  ``xarray.open_zarr("<...>/0000.zarr", group=<ids>)``.

Output is ``<store_path>/<port>/<NNNN>`` per occurrence (``.msgpack`` for raw,
``.zarr`` for distill).

Settings
--------

All optional:

* **store_path**: root for the output. Defaults to the run folder.
* **format**: ``imas`` | ``raw`` | ``distill``. Defaults to ``imas``.
* **per_message** (``imas``): write each message as its own DBEntry
  (``<NNNN>_<seq>``), readable while the run is going. Defaults to ``false``.
* **auto** (``distill``): record every time-dependent 0D/1D/2D ``FLT`` quantity
  (GGD/grid excluded). Defaults to ``true``.
* **config** (``distill``): path to a Python file defining
  ``extract(ids) -> dict[str, xarray.Dataset]`` for derived/geometric
  quantities, recorded alongside (or, with ``auto: false``, instead of) the
  auto-discovered ones.

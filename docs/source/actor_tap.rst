.. _`actor_tap`:

Tap recorder actor
==================

A terminal (sink-only) actor that taps onto an arbitrary number of independent
*timelines* at once and records every message it receives to disk, one DBEntry
per message. It is useful for capturing the live traffic of a running workflow
for later inspection without disturbing the coupling.

.. code-block:: bash

  implementations:
    tap_component:
      executable: python
      args: -u -m imas_muscle3.actors.tap_component

Dynamic ports
-------------

The tap uses MUSCLE3's :ref:`dynamic port configuration <muscle3:Dynamic port
configuration>`: it is created without a fixed port list, so its ports come from
the yMMSL configuration. It accepts **any connected S port whose name is a valid
IDS name** (an optional ``_in`` suffix is stripped, so both ``equilibrium`` and
``equilibrium_in`` are accepted). Each message is deserialized as that IDS.

Each connected S port is a separate timeline, drained by its own thread, so an
idle timeline cannot head-of-line block a busy one. A timeline ends when its
peer sends a message with no ``next_timestamp`` (the usual end-of-stream
convention). The tap is terminal: it has no O_I/O_F/F_INIT ports.

Storage layout
--------------

Each received message is written to its own DBEntry at::

  imas:hdf5?path=<store_path>/<port_name>/<seq>

where ``<seq>`` is a zero-padded, per-timeline sequence number. ``store_path``
defaults to the instance's run folder (its working directory in the MUSCLE3
run), so by default a tap records into its own component directory. The
recordings can be read back with the usual IMAS-Python tooling, e.g.::

  imas.DBEntry("imas:hdf5?path=<store_path>/equilibrium_in/00000000", "r").get("equilibrium")

The full IMAS URI of every recorded message is logged at ``INFO`` level so it
can be copied straight into ``imas.DBEntry(...)`` to reopen that message.

There is no storage cap: the tap records everything it receives.

Backpressure monitoring
-----------------------

A single thread drains all timelines round-robin, so backpressure is a property
of *that* thread rather than any one port. A background monitor logs its
**saturation ratio** ``t_write / (t_wait + t_write)`` — the fraction of loop
time spent recording versus blocked waiting for any port to deliver. A ratio
near 0 means the drain is idle waiting for data; a ratio near 1 means recording
is the bottleneck and the senders are likely stalling on the tap. The monitor
emits a warning when the drain crosses ``saturation_warn``. Per-port message
counts and handler costs are logged too, but only as diagnostics (e.g. to spot
one slow IDS): a per-port wait would just reflect round-robin scheduling, since
a port's next message is usually already buffered by the time the loop returns
to it.

Available Settings
------------------

* Optional

  - **store_path**: (string) Directory root under which per-message DBEntries
    are written. Defaults to the instance's run folder (its working directory).
  - **clean_on_start**: (bool) Remove this tap's own ``<store_path>/<port>``
    subdirectories before recording (never ``store_path`` itself). Defaults to
    ``true``.
  - **monitor_interval**: (float) Seconds between backpressure log lines.
    Defaults to ``5.0``.
  - **saturation_warn**: (float) Drain saturation ratio above which a
    backpressure warning is logged. Defaults to ``0.8``.

Available Ports
---------------

* Optional

  - **<ids_name>[_in] (S)**: Any incoming IDS on the S operator. The port name
    is the IDS name, optionally suffixed with ``_in``.

General
-------
The tap recorder is not bound to a specific DD version.

.. _`actor_iterator`:

Iterator actor
==============

The opposite of the :ref:`accumulator actor <actor_accumulator>`: it takes a
full multi-timeslice IDS received once per IDS name on the F_INIT port,
disassembles it, and sends the individual timeslices out one by one on the
O_I port, chaining ``next_timestamp`` the way a real timestepping source
would. Useful for feeding a serially-stepped actor/simulation from an
already-assembled IDS, e.g. one produced by an accumulator actor upstream.

.. code-block:: yaml

    components:
      iter:
        implementation: iterator
        ports:
          f_init: [equilibrium_in]
          o_i: [equilibrium_out]
    implementations:
      iterator:
        executable: python
        args: -u -m imas_muscle3.actors.iterator_component

With a single connected IDS (as above), ``time_source_ids`` can be left
unset; it's only needed to disambiguate when more than one IDS is
connected.

Available Settings
-------------------

* Optional

  - **time_source_ids** (str): name of the connected input IDS whose time
    range determines the output timeslices. Required when more than one
    IDS is connected on the F_INIT ports; automatically inferred when
    exactly one is connected.
  - **n_timeslices** (int): amount of evenly spaced timeslices spanning the
    full time range of ``time_source_ids``. If not set, that IDS's own
    native time array is used as-is.
  - **interpolation_method** (str): which IMAS interpolation method to use
    when slicing (``closest``, ``previous`` or ``linear``), defaults to
    ``closest``.

Available Ports
----------------
All IDS's are available for the iterator actor. They will be active if connected in the ymmsl file and will be skipped otherwise.

* Optional

  - **<ids_name>_in (F_INIT)**: Any incoming full IDS on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in.
  - **<ids_name>_out (O_I)**: The disassembled timeslices for that IDS, sent one by one on the O_I port. Replace <ids_name> with the required ids i.e. equilibrium_out. Needs to match with incoming port.

General
-------
The iterator actor is not bound to a specific DD version.

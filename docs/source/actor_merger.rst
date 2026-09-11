.. _`actor_merger`:

Merger actor
============

Actor that overlays one IDS on top of another. Every node the overlay has a value for is
written over the base, and everything the overlay is silent about keeps the base's value.
Useful when an actor produces part of an IDS rather than all of it: send whatever the rest
should come from as the base, that actor's output as the overlay, and the downstream actor
receives one complete IDS.

.. code-block:: bash

  implementations:
    merger_component:
      executable: python
      args: -u -m imas_muscle3.actors.merger_component

Available Ports
---------------
All IDS's are available for the merger actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
Each IDS name is handled independently: its base and its overlay are received on F_INIT, merged, and the result is sent on O_F.
All three ports of an IDS must be connected, or none of them.

* Optional

  - **<ids_name>_base (F_INIT)**: The IDS to start from. Replace <ids_name> with the required ids i.e. equilibrium_base.
  - **<ids_name>_overlay (F_INIT)**: The IDS to write over it. Replace <ids_name> with the required ids i.e. equilibrium_overlay.
  - **<ids_name>_out (O_F)**: The merged IDS. Replace <ids_name> with the required ids i.e. equilibrium_out.

An error is raised on startup if an IDS has only some of its three ports connected, or if no IDS is connected at all.

Merging
-------
The merge adheres to the following rules:

* a leaf the overlay has a value for is written over the base's.
* a leaf the overlay leaves empty keeps the base's value.
* an array of structure is merged element by element. An overlay shorter than the base
  leaves the base's remaining elements untouched; an overlay *longer* than the base is an
  error, since merging it would mean inventing the base elements it does not have.

Nothing is interpolated, index ``i`` on one side has to mean the
same instant as index ``i`` on the other. This means both IDSs must be in homogeneous time mode
and share the same root ``/time``, the actor refuses to merge them otherwise. 

The outgoing message carries the base message's timestamps. If the two incoming messages
disagree about their timestamp a warning is logged.

Example
-------
A designer that prescribes only the plasma current and the toroidal field, merged onto the
equilibrium a solver produced:

.. code-block:: yaml

  conduits:
    solver.equilibrium_out: [designer.equilibrium_in, merger.equilibrium_base]
    designer.equilibrium_out: merger.equilibrium_overlay
    merger.equilibrium_out: next_solver.equilibrium_in

The base is a complete equilibrium, on a time base of 3 slices:

.. code-block:: text

  equilibrium
    time                                              [1.0, 2.0, 3.0]
    vacuum_toroidal_field/r0                          6.2
    vacuum_toroidal_field/b0                          [-2.0, -2.0, -2.0]
    time_slice(:)/global_quantities/ip                [1.0e6, 1.1e6, 1.2e6]
    time_slice(:)/global_quantities/psi_boundary      [-104.6, -94.1, -88.2]
    time_slice(:)/boundary/outline/r                  ...
    time_slice(:)/profiles_1d/psi                     ...

The overlay carries only the two quantities the designer owns, on that same time base.
Everything else in it is empty:

.. code-block:: text

  equilibrium
    time                                              [1.0, 2.0, 3.0]
    vacuum_toroidal_field/b0                          [-2.65, -2.65, -2.65]
    time_slice(:)/global_quantities/ip                [-3.0e6, -3.0e6, -3.0e6]

Sent out on ``equilibrium_out`` is the base with those two written over it. ``b0`` and
``ip`` come from the overlay; ``r0``, ``psi_boundary``, the boundary outline and the
profiles are untouched, because the overlay says nothing about them:

.. code-block:: text

  equilibrium
    time                                              [1.0, 2.0, 3.0]
    vacuum_toroidal_field/r0                          6.2                       (base)
    vacuum_toroidal_field/b0                          [-2.65, -2.65, -2.65]     (overlay)
    time_slice(:)/global_quantities/ip                [-3.0e6, -3.0e6, -3.0e6]  (overlay)
    time_slice(:)/global_quantities/psi_boundary      [-104.6, -94.1, -88.2]    (base)
    time_slice(:)/boundary/outline/r                  ...                       (base)
    time_slice(:)/profiles_1d/psi                     ...                       (base)

General
-------
The merger actor is not bound to a specific DD version. Both incoming IDSs are
deserialized against the same Data Dictionary, so they are converted to a common version
if they were sent in different ones.

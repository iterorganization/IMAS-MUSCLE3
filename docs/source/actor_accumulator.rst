.. _`actor_accumulator`:

Accumulator actor
=================

Actor for accumulating timeslices for IMAS data in a simulation into a single IDS.
Useful for running specific actors/simulations serially within a workflow.

Available Ports
---------------
All IDS's are available for the accumulator actor. They will be active if connected in the ymmsl file and will be skipped otherwise.

* Optional
- ***<ids_name>_in (S)***: Any incoming IDS's on the S port. Replace <ids_name> with the required ids i.e. equilibrium_in.
- ***t_next (S)***: Port specifically used to override the next timestep of incomin IDSsand have a centralized stopping condition for the
    accumulation of timesteps. If used, the actor will no longer look at the IDSs themselves but will use the next_timestamp attribute of this port.
- ***<ids_name>_out(O_F)***: Any outgoing IDS's on the O_F port. Replace <ids_name> with the required ids i.e. equilibrium_out. Needs to match with incoming port.

General
-------
The accumulator actor is not bound to a specific DD version.

.. _`actor_passthrough`:

Passthrough actor
=================

Actor that forwards IDS data through unchanged. Useful for bridging workflows or bypassing
another actor without editing the ymmsl coupling.

.. code-block:: bash

  implementations:
    passthrough_component:
      executable: python
      args: -u -m imas_muscle3.actors.passthrough_component

Available Ports
---------------
All IDS's are available for the passthrough actor. They will be active if connected in the ymmsl file and will be skipped otherwise.
Each IDS name is handled independently: any F_INIT message is forwarded once to O_F, and any S messages are forwarded one by one to O_I
until the incoming message's next_timestamp is None. If both an F_INIT and S input are connected for the same IDS and O_F is connected,
the F_INIT message takes precedence over the last received S message.

* Optional

  - **<ids_name>_in_f (F_INIT)**: Any incoming IDS's on the F_INIT port. Replace <ids_name> with the required ids i.e. equilibrium_in_f.
  - **<ids_name>_in_s (S)**: Any incoming IDS's on the S port. Replace <ids_name> with the required ids i.e. equilibrium_in_s.
  - **<ids_name>_out_f (O_F)**: Any outgoing IDS's on the O_F port. Replace <ids_name> with the required ids i.e. equilibrium_out_f. Needs a matching F_INIT or S input.
  - **<ids_name>_out_i (O_I)**: Any outgoing IDS's on the O_I port. Replace <ids_name> with the required ids i.e. equilibrium_out_i. Needs a matching S input.

An error is raised on startup if an output port is connected for an IDS without a matching input port for that same IDS.

General
-------
The passthrough actor is not bound to a specific DD version.

Recording model
---------------

The recorder uses MUSCLE3's :ref:`dynamic port configuration <muscle3:Dynamic
port configuration>`: it is created without a fixed port list, so its ports come
from the yMMSL configuration. It accepts **any connected S port whose name is a
valid IDS name** (an optional ``_in`` suffix is stripped, so both ``equilibrium``
and ``equilibrium_in`` are accepted); each message is deserialized as that IDS.
The recorder is terminal -- it has no O_I/O_F/F_INIT ports.

A single thread drains all timelines round-robin, one outstanding ``receive`` at
a time (libmuscle allows only one pending receive per instance). Each timeline is
drained to its real port close, **not** stopped at the first message with no
``next_timestamp``: under an outer loop the sender ends one reuse with
``next_timestamp=None`` and then sends again, so every iteration is recorded.
Timelines with different message *counts* are fine -- one that ends earlier
simply drops out while the others keep draining. Differing *rates* are the one
caveat: because ``receive`` blocks, a slow timeline holds up the loop while a
fast peer keeps buffering into its (unbounded) send outbox, drained only once
per pass until the slow one closes. This is harmless for the lock-step couplings
these recorders tap, but memory-heavy for a genuinely high-volume, uneven one --
it backs up in the busy sender's memory, never deadlocks or drops messages.

Each timeline ends when its peer's ``ClosePort`` arrives (received as a normal
message). The recorder reads at the communicator level rather than via
``instance.receive``, which would otherwise shut the whole instance down on the
first port's close and sever the other still-draining timelines.

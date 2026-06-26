Recording model
---------------

The recorder uses MUSCLE3 :ref:`dynamic ports <muscle3:Dynamic port
configuration>`: every connected ``S`` port is an independent *timeline* it
records. A port's name is the IDS it carries, optionally with an ``_in`` suffix
(``equilibrium`` and ``equilibrium_in`` are both accepted). The recorder is
terminal — it has only ``S`` ports.

Each timeline is written to disk as its messages arrive — durable, and
tailable while the run is still going — and is recorded in full until that
port's stream closes. A driven sender that re-runs the same grid each
outer-loop iteration is recorded across every iteration; the end of one
iteration is not the end of the timeline.

# Stream ends: `next_timestamp=None` vs. ClosePort

Why the tap/distill recorders drain the way they do
(`_tap_base.serve_timelines`, `DistillHandler`).

A recorder reads an S port in a loop. Two signals look like "the end":

1. **`next_timestamp is None`** is metadata on a normal message: the last
   message of the sender's *current* reuse. The conduit stays open. A sender in
   an outer loop emits it, loops round, and sends again (usually with time reset
   to the start of the grid), so it marks an iteration boundary, not the end.

2. **ClosePort** is the real end, sent once when the sender deregisters. It
   arrives as an ordinary received message whose `data` is a `ClosePort`
   instance, so we detect it with `isinstance(msg.data, ClosePort)`.

## Receive at the communicator level, not `instance.receive()`

We call `instance._communicator.receive_message(port)` rather than
`instance.receive(port)`. The latter, the moment *any one* input receives a
`ClosePort`, calls `Instance.__shutdown()` — libmuscle treats a closed input as
"the whole run is over" — which tears down this instance's connections to
*every* peer. For a multi-timeline tap that drains each port to its own close,
that means the first timeline to end would sever the still-draining others and
drop their undelivered messages. Receiving at the communicator level returns the
`ClosePort` as a normal message with no such side effect; the instance shuts
down only once `serve_timelines` returns and `reuse_instance()` reports done.

(This bypasses `instance.receive`'s MMSF sequence checks too, which is fine — a
terminal tap has no submodel loop to validate.)

## Occurrences

We want one occurrence per outer-loop iteration (`<store>/<port>/<NNNN>.zarr`).
Delivery is not reuse-gated: a single-reuse, S-only tap keeps calling
`receive_message()` and gets every message from all the sender's reuses. So the
recorder drains to the real `ClosePort` (not to `next_timestamp=None`), and
`DistillHandler` rolls a new occurrence at each restart (`next_timestamp=None`,
or message time stepping backward). Stopping at the first `next_timestamp=None`
caught only the first iteration — the original bug.

## Note

Because the sender's `wait_for_receivers` delivers the `ClosePort` before it
closes its server, the recorder receives it cleanly — there is no stall, and no
need for an `F_INIT` trigger or any extra wiring. The remaining libmuscle-side
wart is that `instance.receive` shuts the whole instance down on a single
input's close; until that's fixed upstream, receiving at the communicator level
is the workaround.

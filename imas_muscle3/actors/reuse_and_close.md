# Stream ends: `next_timestamp=None` vs. ClosePort

Why the tap/distill recorders drain the way they do
(`_tap_base.serve_timelines`, `DistillHandler`).

A recorder reads an S port in a loop. Two signals look like "the end":

1. **`next_timestamp is None`** is metadata on a normal message: the last
   message of the sender's *current* reuse. The conduit stays open. A sender in
   an outer loop emits it, loops round, and sends again (usually with time reset
   to the start of the grid), so it marks an iteration boundary, not the end.

2. **ClosePort** is the real end, sent once when the sender deregisters. There
   is no ClosePort message; the next `receive()` raises instead, in one of a few
   teardown-race shapes:
   - `RuntimeError(... "was closed" ... peer crash?)` — a ClosePort arrived;
   - `RuntimeError(... peer 'X' "was lost" ...)` — the connection dropped first;
   - `OSError` (bad file descriptor) — the socket was already gone.

   `serve_timelines` treats all three as end-of-data (matching `OSError` or
   "peer crash"). The wording says crash, but for a terminal observer it is the
   normal end; a real crash surfaces via the manager's exit codes.

## Occurrences

We want one occurrence per outer-loop iteration (`<store>/<port>/<NNNN>.zarr`).
Delivery is not reuse-gated: a single-reuse, S-only tap keeps calling
`receive()` and gets every message from all the sender's reuses. So the recorder
drains to the real close (not to `next_timestamp=None`), and `DistillHandler`
rolls a new occurrence at each restart (`next_timestamp=None`, or message time
stepping backward). Stopping at the first `next_timestamp=None` caught only the
first iteration — the original bug.

## Cost

The extra `receive()` after the last iteration hits the gone peer, and
libmuscle's TCP client retries it for a hardcoded `RECONNECT_TIMEOUT = 60.0`s
before raising, so shutdown stalls ~60s (the data is already written; only the
exit is delayed). The clean alternative, `reuse_instance()` returning `False`,
works only with a connected F_INIT port, so it would need an extra `trigger_in`
pulsed per iteration by the driver. We took the no-wiring drain instead.

A future muscle3 is expected to send a real ClosePort on O_I timelines, which
would end the drain cleanly and drop the 60s stall.

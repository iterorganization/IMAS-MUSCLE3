# MUSCLE3 stream ends: `next_timestamp=None` vs. `ClosePort`

Notes for the tap/distill recorders (`_tap_base.serve_timelines`,
`distill_component`) on how a recorded timeline actually *ends*, and why the
recorder drains the way it does. This is subtle and cost us a long debugging
session, so it's written down here.

## Two different "ends"

A recorder taps a port as an **S** (state) input and just calls
`instance.receive(port)` in a loop. The sender (e.g. TORAX on `…_out_i`) writes
to that conduit. There are **two** distinct end-of-something signals, and they
are easy to conflate:

1. **`next_timestamp is None` — the "intermediate close".**
   This is *metadata on a normal message*: "this is the last message of the
   sender's **current reuse**". It does **not** close the conduit. When the
   sender is inside an outer loop (e.g. a Picard iteration) it finishes one
   reuse — emitting a final message with `next_timestamp=None` — and then goes
   round its **own reuse loop** and starts sending again. So `next_timestamp=
   None` marks an *iteration boundary*, not the end of the data. Think of it as
   an "intermediate ClosePort": the stream pauses and restarts, usually with the
   time axis reset to the start of the grid.

2. **`ClosePort` — the real close.**
   Sent once, when the sender **deregisters for good** (its whole run is over).
   The receiver does not get a `ClosePort` *message*; instead the next
   `instance.receive()` **raises**, in one of a few teardown-race-dependent
   shapes:
   - `RuntimeError("Port … was closed while trying to receive on it, did the
     peer crash?")` — clean case, a `ClosePort` was delivered;
   - `RuntimeError("Error while receiving a message: connection with peer 'X'
     was lost. Did the peer crash?")` — the connection dropped first; or
   - `OSError` (e.g. `Bad file descriptor`) — the peer's socket is already torn
     down by the time we receive.

   All mean "no more data, ever" (`serve_timelines` matches `OSError` or a
   message containing "was closed" / "was lost"). (libmuscle phrases the RuntimeError as a
   possible crash, but for a terminal observer it is simply the normal end; a
   genuine peer crash is reported separately by the manager via exit codes.)

## Why this matters for occurrences

We want **one occurrence per outer-loop iteration**
(`<store>/<port>/<NNNN>.zarr`). Because delivery is **not** reuse-gated — a
single-reuse, S-only tap can keep calling `receive()` and get every message
from *all* of the sender's reuses (proven empirically) — the recorder:

* **drains to the real close**, not to `next_timestamp=None`; and
* **rolls to a new occurrence** at each *intermediate close*
  (`next_timestamp is None`, or a backward step in message time — the grid
  resetting), in `DistillHandler`.

Earlier the recorder stopped at the first `next_timestamp=None`, so it only ever
captured the **first** iteration (one occurrence). That was the bug.

## The cost we accept (for now)

Detecting the *real* close means calling `receive()` once more after the last
iteration and hitting the closed peer. libmuscle's TCP client then retries the
gone peer for a **hardcoded `RECONNECT_TIMEOUT = 60.0` s** (per port, not
configurable) before raising. So a recorder stalls up to ~60 s at shutdown — the
data is already on disk by then, it just delays the instance's exit. We accept
this for now.

## The clean alternative (not taken yet)

MUSCLE3's blessed end-of-stream detector is `reuse_instance()` returning
`False`, which it does cleanly (no 60 s retry) **only for an instance with a
connected `F_INIT` port**. An S-only instance runs exactly one reuse
(`Instance._decide_reuse_instance`: `if not f_init_connected: return
self._first_run`). So the stall-free design is to give the recorder an
**optional `F_INIT` `trigger_in`** pulsed once per iteration by the driver
(like `visualization_component`), and let `reuse_instance()` both drive the
occurrence count and detect the end. That needs one extra port + one conduit in
the workflow; we chose the no-wiring drain-to-close instead, trading a ~60 s
shutdown stall for zero wiring.

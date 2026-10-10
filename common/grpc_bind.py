"""Bind a gRPC server's port, retrying while the previous holder releases it.

Every server in a sweep cell -- the datacentres, the provisioner, the
schedulers -- takes a fixed port that the previous cell's server of the same
role held seconds earlier. The sweep checks that the port is free first, but it
can only ask whether the port ACCEPTS, and a socket in TIME_WAIT refuses
connections while still blocking bind(). The guard passes, add_insecure_port
raises, the server dies, and the sweep reports "never opened port" as though
the start were slow.

120 s of retries, twice the length of TIME_WAIT on Linux. The sweep's own
guard stays a connect test, so a port held by a process that is really still
serving is caught there and aborts immediately; what reaches here is the case
that resolves itself.

**The sweep's readiness wait must outlast this window.** It was 60 s against a
60 s retry for the schedulers and the provisioner, so a server that spent its
budget retrying was killed at the very moment it would have succeeded: one cell
of a 170-cell poll sweep died that way on 2026-09-12 ("scheduler sched-asia
never opened port 50202"). Those waits are now 150 s (the datacentres were
already 120 s).

2026-09-09: 20 cells of a 123-cell TP=2 sweep were lost in the scheduler's bind,
every one an ILP policy, whose teardown carries a solver subprocess and frees its
port late. The retry went into the scheduler alone. 2026-09-10: the Mixtral
grid lost a datacentre (US_MISO, round_robin at 400 req/s) and the provisioner
(carbon_aware at 2000) with the same "never opened port" symptom, which is why
it now lives here and every server uses it.
"""
import logging
import time

log = logging.getLogger(__name__)


def bind_with_retry(server, port, tries=240, gap_s=0.5):
    """add_insecure_port on [::]:port, retried for tries x gap_s seconds."""
    for i in range(tries):
        try:
            server.add_insecure_port(f"[::]:{port}")
            if i:
                log.warning("bound port %d after %d retries (%.1fs); the "
                            "previous holder was slow to release it",
                            port, i, i * gap_s)
            return
        except RuntimeError:
            if i == tries - 1:
                raise
            time.sleep(gap_s)

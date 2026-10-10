"""Build and solve the batch assignment in an interpreter of its own.

**Why this exists, measured rather than assumed.** The batch policies build
their cost matrix in a nested Python loop -- one `_check_sla` and one
`_estimate_energy_j` per (request, datacentre) cell -- so a 300-request batch
over 16 sites is 4,800 interpreter-bound calls. That thread shares its GIL
with the scheduler's event loop, and at 4800 req/s the loop alone wants ~96%
of a core while the build wants ~52%. 148% does not fit in one interpreter, so
one of them starves.

The signature of the starvation is that solve time stopped being monotonic in
batch size: in one run a 1290-request batch took 555 ms (27 us per cell) while
a 301-request batch took 1491 ms (310 us per cell). Problem complexity cannot
produce an 11x swing in per-cell speed within a single run; contention can. The
oversized batches were the SYMPTOM -- a starved solve overruns its window, the
buffer grows, the next solve is bigger -- and 26.5% of ilp_carbon_sla's
requests at 4800 were returned unplaced because the wait outran the handler's
5 s buffer.

**This is a move, not a rewrite.** Vectorising the build would be faster still,
but `_check_sla` and `_estimate_energy_j` ARE the calibrated TTFT/TPOT models;
rewriting them risks the calibration rather than one run. Nothing here touches
a formula. The worker constructs the same policy from the same arguments and
calls the same `choose_dc_batch`, so for a given input the answer is identical
-- which is a testable claim, not a hope.

**spawn, not fork.** The scheduler has live threads and an event loop; forking
copies their locks in whatever state they happened to be in. A spawned child
pays one interpreter start and one model load, once, at startup.

**Failure is a fallback, not an outage.** If the worker dies or overruns, the
caller solves in-process exactly as before. A batch policy that cannot reach
its solver must degrade, not stop.

**A Pipe, not a Queue, and that was measured too.** `mp.Queue.put` is
asynchronous: it hands the payload to a background FEEDER THREAD that pickles
it and writes to the pipe. That thread queues for the same GIL the offload
exists to get away from, so the first version simply moved the contention. The
split timing showed it plainly -- a 302-request batch where the worker spent
84.9 ms and the caller waited 679.6 ms, against 800-request batches where the
worker spent 500 ms and the caller waited 2 ms. The wait tracked the parent's
scheduling, not the problem. `Connection.send` pickles inline in the calling
thread, which is a thread already allowed to block, and `recv` blocks in C
with the GIL released.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

log = logging.getLogger("scheduler.batch_worker")

#: Sent to the worker to make it exit.
_STOP = "__stop__"


class BatchSolverUnavailable(Exception):
    """The worker could not answer; the caller should solve in-process."""


def _build_policy(spec: dict):
    """Reconstruct the scheduler's policy inside the worker.

    Built from arguments rather than shipped as an object: the analytical model
    holds fitted tables and is not something to pickle per batch. The cost is
    one construction at startup.
    """
    from scheduler.policies import get_policy

    kwargs = dict(spec.get("policy_kwargs") or {})
    model_name = spec.get("model_name") or ""
    if model_name:
        from datacenter.perf_models.analytical.latency import (
            AnalyticalLatencyModel,
        )
        kwargs["analytical_model"] = AnalyticalLatencyModel(
            model_name, tp=spec.get("tp_size", 1))
    return get_policy(
        spec["policy_name"],
        alpha=spec.get("alpha", 0.0),
        scheduler_group=spec["scheduler_group"],
        **kwargs,
    )


def _die_with_parent():
    """Ask the kernel to kill this process when its parent goes.

    Belt and braces over the pipe's EOF, and the braces were needed: the first
    version of this module used mp.Queue, whose `get()` does NOT end when the
    writer dies, so a scheduler killed with SIGKILL left its solver alive
    forever -- `daemon=True` does not help, because the parent's atexit handler
    is exactly what SIGKILL skips. Two such orphans were found still resident
    hours later on 2026-09-07, and repeated killed runs would accumulate them.

    Linux only, and silently skipped elsewhere; the pipe EOF still covers the
    ordinary case.
    """
    try:
        import ctypes
        import signal
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:                                            # noqa: BLE001
        return
    # The parent can die between the fork and the prctl above, and the signal
    # is then never sent. Reparenting to init is what that looks like.
    if os.getppid() == 1:
        os._exit(0)


def _worker_main(spec, conn):
    """Answer solve requests until told to stop. Runs in the child process."""
    logging.basicConfig(level=logging.WARNING)
    _die_with_parent()
    import inference_pb2
    from scheduler.policies.common.estimators import RequestInfo

    init_error = None
    try:
        policy = _build_policy(spec)
    except Exception as e:                                       # noqa: BLE001
        # Answer every job with the same failure, so the parent falls back
        # cleanly instead of blocking on a worker that will never reply.
        init_error = f"worker init failed: {e}"
        policy = None

    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job == _STOP:
            return
        job_id, reqs, dc_bytes, region_id, rtt_matrix, max_bs, t_sent = job
        # Same host, same clock, so this is a real one-way transit time: how
        # long the job spent between the parent handing it to the pipe and this
        # process being scheduled to read it.
        t_recv = time.monotonic()
        if policy is None:
            conn.send((job_id, {"__error__": init_error or "no policy"}, {}))
            continue
        try:
            req_infos = [RequestInfo(r[0], r[1], r[2], r[3]) for r in reqs]
            dc_view = {k: inference_pb2.DcStats.FromString(v)
                       for k, v in dc_bytes.items()}
            t0 = time.monotonic()
            out = policy.choose_dc_batch(
                req_infos, dc_view, region_id,
                max_batch_size=max_bs, rtt_matrix=rtt_matrix,
            )
            # The worker's OWN elapsed time, reported alongside the answer.
            # Without it the parent measures a round trip and cannot tell a
            # slow solve from a parent that could not get scheduled to read
            # the reply -- and at 96% loop CPU the second is the likelier of
            # the two. Two causes in one number is how the admit gap, the
            # datacentre queue and the batch cadence each hid.
            t_end = time.monotonic()
            conn.send((job_id, out, {
                "worker_ms": (t_end - t0) * 1000.0,
                # Deserialising the payload, before any solving.
                "unpack_ms": (t0 - t_recv) * 1000.0,
                "in_ms": (t_recv - t_sent) * 1000.0,
                "t_reply": t_end,
            }))
        except Exception as e:                                   # noqa: BLE001
            conn.send((job_id, {"__error__": f"{type(e).__name__}: {e}"}, {}))


class BatchSolver:
    """A persistent solver process, addressed one batch at a time."""

    def __init__(self, spec: dict) -> None:
        self._spec = spec
        self._ctx = mp.get_context("spawn")
        self._conn, self._child_conn = self._ctx.Pipe(duplex=True)
        self._proc: Optional[mp.process.BaseProcess] = None
        self._job_id = 0
        #: Milliseconds the worker itself spent on the last batch, as opposed
        #: to the round trip the caller timed.
        self.last_worker_ms = 0.0
        #: The round trip, split. Three hypotheses about where the caller's
        #: wait went were wrong in a row -- the build contending for the GIL
        #: (partly), an mp.Queue feeder thread (no), and GIL hand-off latency
        #: (no: a 0.5 ms switch interval left a 794 ms wait at 832 ms). So the
        #: transit is timed directly instead: `in` is parent-send to
        #: child-receive, `unpack` is the child deserialising, `work` is the
        #: solve, `out` is child-reply to parent-receive. Same host, same
        #: monotonic clock, so the one-way times are real.
        self.last_split = {}

    def start(self) -> None:
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._spec, self._child_conn),
            daemon=True,
            name="ilp-solver",
        )
        self._proc.start()
        # The child holds its own end; keeping a copy here would mean recv()
        # never sees EOF if the worker exits.
        self._child_conn.close()
        log.info("batch solver process started (pid=%s)", self._proc.pid)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _payload(self, job_id, requests, dc_view, region_id, rtt_matrix,
                 max_batch_size):
        """What crosses the pipe: tuples and serialised protobuf. Explicit,
        small (a 300-request batch over 16 sites is well under 100 KB), and not
        dependent on a class staying picklable."""
        return (
            job_id,
            [(r.req_id, r.in_tok, r.out_tok, r.predicted_osl)
             for r in requests],
            {k: v.SerializeToString() for k, v in dc_view.items()},
            region_id,
            dict(rtt_matrix or {}),
            max_batch_size,
            time.monotonic(),
        )

    def _record_timing(self, timing, t_got):
        self.last_worker_ms = float(timing.get("worker_ms", 0.0))
        self.last_split = {
            "in": float(timing.get("in_ms", 0.0)),
            "unpack": float(timing.get("unpack_ms", 0.0)),
            "work": self.last_worker_ms,
            "out": ((t_got - timing["t_reply"]) * 1000.0
                    if "t_reply" in timing else 0.0),
        }

    def solve(self, requests: List, dc_view: Dict, region_id: str,
              rtt_matrix: Dict, max_batch_size: int = 0,
              timeout: float = 10.0) -> Dict[str, str]:
        """Solve one batch. Raises BatchSolverUnavailable rather than blocking.

        Requests cross as tuples and datacentres as serialised protobuf: both
        are explicit and small (a 300-request batch over 16 sites is well under
        100 KB), and neither depends on a class staying picklable.
        """
        if not self.alive():
            raise BatchSolverUnavailable("solver process is not running")

        self._job_id += 1
        job_id = self._job_id
        payload = self._payload(job_id, requests, dc_view, region_id,
                                rtt_matrix, max_batch_size)
        try:
            # Pickles and writes in THIS thread. mp.Queue would hand it to a
            # feeder thread that then queues for the GIL -- see the note at the
            # top of this module.
            self._conn.send(payload)
        except (BrokenPipeError, OSError) as e:
            raise BatchSolverUnavailable(f"could not reach the solver: {e}")

        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise BatchSolverUnavailable(
                    f"solver did not answer within {timeout:.1f}s")
            if not self._conn.poll(min(left, 0.5)):
                if not self.alive():
                    raise BatchSolverUnavailable("solver process died")
                continue
            try:
                got_id, out, timing = self._conn.recv()
                t_got = time.monotonic()
            except EOFError:
                raise BatchSolverUnavailable("solver closed the connection")
            if got_id != job_id:
                # A reply to a batch the caller already gave up on. Dropping it
                # is right -- those requests were placed by the in-process
                # fallback -- but it must not be handed to THIS batch, which
                # would assign one window's decisions to another's requests.
                log.warning("discarding a reply for batch %s while waiting "
                            "for %s", got_id, job_id)
                continue
            if isinstance(out, dict) and "__error__" in out:
                raise BatchSolverUnavailable(out["__error__"])
            self._record_timing(timing, t_got)
            return out

    async def solve_async(self, requests: List, dc_view: Dict, region_id: str,
                          rtt_matrix: Dict, max_batch_size: int = 0,
                          timeout: float = 10.0) -> Dict[str, str]:
        """`solve`, with the reply collected by the event loop.

        The reason this exists. At 4800 req/s, the round trip split four ways
        over 72 single-chunk cycles:

            in      0.3 ms   parent handed it over -> child read it
            unpack  0.1 ms   child deserialising
            work   77.0 ms   the solve
            out   833.2 ms   child replied -> parent read it

        Nothing is slow except the collection. The reply sits in the pipe while
        the batch thread, blocked in poll(), waits to be scheduled. A 0.5 ms
        GIL switch interval did not move it (794 ms became 832 ms), so it is
        not a shortage of hand-off opportunities -- though that does not fully
        acquit the GIL, since a waiting thread can lose every hand-off race it
        is offered.

        Awaiting on the loop makes the question moot: `add_reader` gives the fd
        to a thread that is already running, so there is nothing to wake.
        """
        if not self.alive():
            raise BatchSolverUnavailable("solver process is not running")

        self._job_id += 1
        job_id = self._job_id
        payload = self._payload(job_id, requests, dc_view, region_id,
                                rtt_matrix, max_batch_size)
        try:
            self._conn.send(payload)
        except (BrokenPipeError, OSError) as e:
            raise BatchSolverUnavailable(f"could not reach the solver: {e}")

        loop = asyncio.get_running_loop()
        fd = self._conn.fileno()
        deadline = loop.time() + timeout
        while True:
            waiter = loop.create_future()
            loop.add_reader(fd, lambda: (not waiter.done()
                                         and waiter.set_result(None)))
            try:
                await asyncio.wait_for(waiter,
                                       max(0.0, deadline - loop.time()))
            except asyncio.TimeoutError:
                raise BatchSolverUnavailable(
                    f"solver did not answer within {timeout:.1f}s")
            finally:
                loop.remove_reader(fd)

            try:
                got_id, out, timing = self._conn.recv()
            except EOFError:
                raise BatchSolverUnavailable("solver closed the connection")
            t_got = time.monotonic()
            if got_id != job_id:
                # A reply to a batch the caller already gave up on. Dropping it
                # is right -- those requests went to the in-process fallback --
                # but it must not be given to THIS batch, which would assign
                # one window's decisions to another window's requests.
                log.warning("discarding a reply for batch %s while waiting "
                            "for %s", got_id, job_id)
                continue
            if isinstance(out, dict) and "__error__" in out:
                raise BatchSolverUnavailable(out["__error__"])
            self._record_timing(timing, t_got)
            return out

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._conn.send(_STOP)
            self._proc.join(timeout=5.0)
        except Exception:                                        # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.terminate()

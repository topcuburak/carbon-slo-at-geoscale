"""Many requests over one gRPC stream, behind the unary stub's interface.

The scheduler spends most of its CPU in the gRPC runtime rather than in
scheduling -- measured 176.8 us of server CPU per unary call against 90.7 us
per message on a stream, a 2.24x difference that is per-call HTTP/2 stream
setup and teardown. This is the client half of removing that cost for the
load generator's hop.

`StreamMux` exposes exactly the one method of a stub that the generator uses,
with the same signature and the same semantics, so `send_one_request` cannot
tell the difference and the two transports stay measurable against each other.
Everything else here exists to make one stream behave like N independent calls:

  * Replies arrive in COMPLETION order, so the response has to be routed back
    to the coroutine that asked for it. That is what `_pending` is: a request
    id to the future waiting on it. Getting this wrong would not raise -- it
    would attribute one request's dc_id, service_ms and carbon_intensity to
    another request, silently, in every row of the results file. So an id that
    is not in the map is an error and says so, and the reply is dropped rather
    than given to whoever happens to be next.
  * A single stream is a single failure domain. When it breaks, every request
    waiting on it must be failed rather than left to hang until the run's
    drain timeout, which would look like a fleet stall.

Whether a broken stream is re-established is the caller's choice, and the two
callers want opposite things. For the load generator a broken stream means the
run is void, so it stays broken and says so. For a scheduler forwarding to a
peer it must NOT: the unary path recovers on its own through the channel, so a
mux that failed permanently would turn a momentary blip into a quarter of the
fleet unable to forward for the rest of the run -- a behaviour change dressed
up as an optimisation. That is what `reconnect` is for.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Set


import inference_pb2
import inference_pb2_grpc

log = logging.getLogger("loadgen.stream_mux")


class StreamClosed(Exception):
    """The stream carrying this request went away before it was answered."""


class StreamMux:
    """One SubmitStream to one scheduler, shared by every request to it."""

    def __init__(self, stub: inference_pb2_grpc.SchedulerServiceStub,
                 sched_id: str, reconnect: bool = False) -> None:
        self._stub = stub
        self._sched_id = sched_id
        self._reconnect = reconnect
        self._pending: Dict[str, asyncio.Future] = {}
        # Ids whose caller has already given up. A reply for one of these is
        # expected and uninteresting; a reply for anything else is a routing
        # bug and must not be quiet. Bounded by the number of timeouts, which
        # is zero in a healthy run.
        self._abandoned: Set[str] = set()
        self._outbox: asyncio.Queue = asyncio.Queue()
        # How long each request sat in the outbox before it went on the wire.
        # This is a NEW queue that no existing column covers: SubmitRequest
        # returns the moment it enqueues, so if the stream's send window fills,
        # requests pile up here and the delay lands in wall_ms with nothing to
        # attribute it to -- the same shape as the `admit` gap that hid a
        # thread-pool wall for a week. pool_wait_ms does not cover it; that is
        # event-loop lag before the request is ever built.
        self._queued_at: Dict[str, float] = {}
        self.outbox_ms: Dict[str, float] = {}
        self._pump: Optional[asyncio.Task] = None
        self._failure: Optional[BaseException] = None
        self._closed = False
        self._done = object()

    def start(self) -> None:
        self._pump = asyncio.create_task(self._run())

    def _ensure_pump(self) -> None:
        """Re-establish the stream after a failure, when the caller allows it.

        Requests already sitting in the outbox are picked up by the new pump,
        so a blip costs the in-flight requests and nothing after them.
        """
        if self._pump is None or self._pump.done():
            self._failure = None
            self._closed = False
            self._pump = asyncio.create_task(self._run())

    async def _outgoing(self):
        while True:
            msg = await self._outbox.get()
            if msg is self._done:
                return
            t_q = self._queued_at.pop(msg.req_id, None)
            if t_q is not None:
                self.outbox_ms[msg.req_id] = (time.monotonic() - t_q) * 1000.0
            yield msg

    async def _run(self) -> None:
        try:
            async for resp in self._stub.SubmitStream(self._outgoing()):
                fut = self._pending.pop(resp.req_id, None)
                if fut is None:
                    if resp.req_id in self._abandoned:
                        self._abandoned.discard(resp.req_id)
                    else:
                        log.error(
                            "%s: reply for req_id=%r that nothing is waiting "
                            "for. A response has been routed to the wrong "
                            "request or duplicated; the run's per-request "
                            "columns cannot be trusted.",
                            self._sched_id, resp.req_id)
                    continue
                if not fut.done():
                    fut.set_result(resp)
        except asyncio.CancelledError:
            self._fail_all(StreamClosed("stream cancelled"))
            raise
        except Exception as e:                                   # noqa: BLE001
            log.error("%s: SubmitStream failed: %s", self._sched_id, e)
            self._fail_all(e)
        else:
            # The server closed its side. Anything still waiting never will be.
            self._fail_all(StreamClosed("server closed the stream"))

    def _fail_all(self, exc: BaseException) -> None:
        self._failure = exc
        for req_id, fut in list(self._pending.items()):
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.set_exception(exc)

    async def SubmitRequest(self, request: inference_pb2.InferenceRequest,
                            timeout: float = 2000.0):
        """Send one request and wait for the reply that carries its req_id.

        Same contract as the unary stub method, including raising on timeout,
        so the caller's error handling is unchanged.
        """
        if self._reconnect and not self._closed:
            self._ensure_pump()
        if self._failure is not None:
            raise self._failure
        if not request.req_id:
            # Without an id there is nothing to match a reply against. The
            # unary path tolerates this and lets the scheduler mint one; here
            # it would mean the reply could never be delivered.
            raise ValueError("a streamed request must carry a req_id")

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request.req_id] = fut
        self._queued_at[request.req_id] = time.monotonic()
        await self._outbox.put(request)
        try:
            return await asyncio.wait_for(fut, timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Drop the future rather than leaving it in the map for a reply
            # that may still be in flight -- the map is per-run and a leak here
            # is a leak for the whole run.
            if self._pending.pop(request.req_id, None) is not None:
                self._abandoned.add(request.req_id)
            self._queued_at.pop(request.req_id, None)
            raise

    def pop_outbox_ms(self, req_id: str) -> float:
        """Milliseconds this request waited in the outbox. Popped, not read:
        the caller writes it to its row and the entry must not outlive it."""
        self._queued_at.pop(req_id, None)
        return self.outbox_ms.pop(req_id, 0.0)

    async def close(self) -> None:
        self._closed = True
        await self._outbox.put(self._done)
        if self._pump is not None:
            try:
                await asyncio.wait_for(self._pump, timeout=30.0)
            except Exception:                                    # noqa: BLE001
                self._pump.cancel()


def build_scheduler_streams(stubs: Dict[str, inference_pb2_grpc.SchedulerServiceStub]
                            ) -> Dict[str, StreamMux]:
    """Wrap every stub in a mux and start it. Same keys, same call surface."""
    muxes = {sid: StreamMux(stub, sid) for sid, stub in stubs.items()}
    for m in muxes.values():
        m.start()
    return muxes

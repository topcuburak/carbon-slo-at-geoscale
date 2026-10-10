"""Call the scheduler's coroutine handlers from a synchronous test.

SchedulerServicer's handlers became coroutines when the server moved to
grpc.aio. The tests are synchronous and there are roughly thirty call sites
between test_scheduler.py and test_forwarding.py, so the wrapping happens once
here rather than at each of them:

  * SubmitRequest and PushSnapshot run to completion on a private loop, so a
    test still writes `r = s.SubmitRequest(req, None)`.
  * A test that replaces `_dispatch_to_dc` with a plain function still works.
    The server awaits that attribute now; a plain function is adapted into a
    coroutine on the way in, so the fakes stay readable.

The private loop is also handed to the servicer as `_loop`, which is what the
ILP batch thread uses to wake a buffered request -- without it, a batch-policy
test would block until its 5 s timeout.
"""

import asyncio
import inspect


class SyncServicer:
    """A SchedulerServicer that answers synchronously."""

    #: Run on the loop instead of handing back a coroutine.
    _HANDLERS = ("SubmitRequest", "PushSnapshot")
    #: Attributes the server awaits: a test's plain function needs adapting.
    _AWAITED = ("_dispatch_to_dc",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        loop = asyncio.new_event_loop()
        object.__setattr__(self, "_test_loop", loop)
        inner._loop = loop

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._HANDLERS:
            def _run(*args, **kwargs):
                return self._test_loop.run_until_complete(attr(*args, **kwargs))
            return _run
        return attr

    def submit_concurrently(self, requests):
        """Submit several requests at once, the way the server now sees them.

        A test that wants concurrent arrivals used to start one thread per
        request. Under grpc.aio concurrent requests are concurrent coroutines
        on ONE loop, and six threads driving a single loop is not a thing --
        so the concurrency is expressed the way production has it. Returns the
        responses in the order the requests were given.
        """
        async def _all():
            return await asyncio.gather(*[
                self._inner.SubmitRequest(r, None) for r in requests])

        return self._test_loop.run_until_complete(_all())

    def start_batch(self):
        """Start the batch loop on the private loop.

        It is a task rather than a thread now, so it needs a loop to be
        created on -- and it makes progress whenever that loop runs, which is
        during every submit_concurrently. Defined here rather than forwarded:
        `__getattr__` would call it with no running loop and it would raise.
        """
        async def _go():
            self._inner.start_batch()
        self._test_loop.run_until_complete(_go())

    def stream(self, requests):
        """Drive SubmitStream to exhaustion and return the replies it wrote.

        In the order the handler produced them, NOT the order the requests were
        given -- the point of the stream is that those two differ, so a test
        that wants them lined up matches on req_id itself.
        """
        async def _run():
            async def _feed():
                for r in requests:
                    yield r

            return [resp async for resp in self._inner.SubmitStream(_feed(), None)]

        return self._test_loop.run_until_complete(_run())

    def __setattr__(self, name, value):
        if name in self._AWAITED and not inspect.iscoroutinefunction(value):
            plain = value

            async def _adapted(*args, **kwargs):
                return plain(*args, **kwargs)

            value = _adapted
        setattr(self._inner, name, value)

#!/usr/bin/env python3
"""The load generator, part by part.

The generator is the measuring instrument: if it does not deliver the arrival
process it claims, every latency in the paper is against the wrong x-axis. So
these check the process itself as well as the plumbing --

  the arrival schedule   is it really Poisson, at really the rate asked for
  reproducibility        one seed, one workload, exactly
  the trace              parsed, cycled deterministically, ground truth kept
                         apart from what a predictor said
  the row                every request writes the same columns, success or not

The scheduler is stubbed, so nothing here needs a fleet or a port.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_loadgen.py
"""

import asyncio
import csv
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

import inference_pb2                                              # noqa: E402
from loadgen.arrival import DISTRIBUTIONS, make_arrival_generator  # noqa: E402
from loadgen.trace import (InferenceRecord, load_trace,           # noqa: E402
                           make_trace_generator)
from loadgen.client import send_one_request as _send_one, run     # noqa: E402


def send_one_request(*args, **kwargs):
    """Drive the coroutine to completion, so the assertions stay synchronous.

    send_one_request is `async def` since the generator moved to grpc.aio.
    Wrapping it here rather than rewriting every check keeps each test about
    what it was about.
    """
    return asyncio.run(_send_one(*args, **kwargs))

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:                                    # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


class FakeSchedStub:
    """A scheduler that answers, fails, or takes its time."""

    def __init__(self, delay=0.0, raises=None, dc_id="US_PJM",
                 sched_path="", **fields):
        self.delay, self.raises, self.dc_id = delay, raises, dc_id
        self.sched_path, self.fields = sched_path, fields
        self.seen = []

    async def SubmitRequest(self, request, timeout=None):
        # async, because the client is grpc.aio now: a request is a coroutine
        # on one event loop, not a thread. `delay` sleeps on the loop for the
        # same reason -- a blocking sleep here would stall every other
        # outstanding request instead of just this one.
        self.seen.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        return inference_pb2.InferenceResponse(
            req_id=request.req_id, dc_id=self.dc_id, status="OK",
            sched_path=self.sched_path, **self.fields)


def write_trace(rows, header=None):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    header = header or ["sample_id", "in_tokens", "out_tokens", "dataset"]
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    f.close()
    return f.name


# ─────────────────────────── the arrival process ────────────────────────────

@check("test_poisson_delivers_the_rate_it_was_asked_for")
def _():
    random.seed(1)
    gen = make_arrival_generator("poisson", 0, 0, arrival_rate=40.0)
    gaps = [gen() for _ in range(200_000)]
    got = 1.0 / statistics.mean(gaps)
    assert abs(got - 40.0) / 40.0 < 0.01, f"asked 40 req/s, got {got:.2f}"


@check("test_poisson_gaps_are_actually_exponential")
def _():
    # Rate alone does not make a process Poisson -- a constant 25 ms gap has
    # the same mean. The distinguishing property is that the standard
    # deviation equals the mean.
    random.seed(2)
    gen = make_arrival_generator("poisson", 0, 0, arrival_rate=40.0)
    gaps = [gen() for _ in range(200_000)]
    mean, sd = statistics.mean(gaps), statistics.stdev(gaps)
    assert abs(sd - mean) / mean < 0.02, \
        f"sd/mean = {sd / mean:.3f}; exponential requires 1.0"
    # and the memoryless tail: P(X > t) = exp(-lambda t)
    tail = sum(1 for g in gaps if g > 1.0 / 40.0) / len(gaps)
    assert abs(tail - math.exp(-1)) < 0.01, tail


@check("test_uniform_stays_inside_its_bounds_and_centres_between_them")
def _():
    random.seed(3)
    gen = make_arrival_generator("uniform", 30.0, 120.0, arrival_rate=0)
    gaps = [gen() for _ in range(100_000)]
    assert min(gaps) >= 0.030 - 1e-9, min(gaps)
    assert max(gaps) <= 0.120 + 1e-9, max(gaps)
    assert abs(statistics.mean(gaps) - 0.075) < 0.001, statistics.mean(gaps)


@check("test_the_same_seed_gives_the_same_arrivals")
def _():
    def draw():
        random.seed(42)
        g = make_arrival_generator("poisson", 0, 0, 40.0)
        return [g() for _ in range(1000)]
    assert draw() == draw(), "the arrival schedule is not reproducible"


@check("test_a_different_seed_gives_different_arrivals")
def _():
    # The negative control for the test above: without it, a generator that
    # returned a constant would pass reproducibility.
    random.seed(42)
    g = make_arrival_generator("poisson", 0, 0, 40.0)
    a = [g() for _ in range(1000)]
    random.seed(43)
    g = make_arrival_generator("poisson", 0, 0, 40.0)
    assert [g() for _ in range(1000)] != a


@check("test_an_unknown_distribution_is_refused_by_name")
def _():
    try:
        make_arrival_generator("gaussian", 0, 0, 40.0)
    except ValueError as e:
        assert "gaussian" in str(e), e
        assert all(d in str(e) for d in DISTRIBUTIONS), \
            "the error does not say what is available"
        return
    raise AssertionError("an unknown distribution was accepted")


# ──────────────────────────────── the trace ─────────────────────────────────

@check("test_a_trace_is_read_with_its_columns_intact")
def _():
    path = write_trace([
        {"sample_id": 1, "in_tokens": 100, "out_tokens": 50, "dataset": "cnn"},
        {"sample_id": 2, "in_tokens": 900, "out_tokens": 20, "dataset": "gsm8k"},
    ])
    recs = load_trace(path)
    assert len(recs) == 2, len(recs)
    assert recs[0] == InferenceRecord(1, 100, 50, "cnn", 0), recs[0]
    assert recs[1].dataset == "gsm8k", recs[1]


@check("test_the_older_column_spelling_still_reads")
def _():
    # The measurement era wrote in_tok/out_tok; the replay era writes
    # in_tokens/out_tokens. Both traces are in the repo.
    path = write_trace([{"sample_id": 7, "in_tok": 64, "out_tok": 8,
                         "dataset": "mbpp"}],
                       header=["sample_id", "in_tok", "out_tok", "dataset"])
    r = load_trace(path)[0]
    assert (r.in_tokens, r.out_tokens) == (64, 8), r


@check("test_a_predicted_length_never_overwrites_the_truth")
def _():
    # out_tokens is what the datacentre will generate; predicted_osl is what a
    # predictor guessed. Confusing them is how a run configured for
    # "predicted" gets scored on oracle knowledge.
    path = write_trace(
        [{"sample_id": 1, "in_tokens": 100, "out_tokens": 500,
          "dataset": "cnn", "predicted_osl": 123}],
        header=["sample_id", "in_tokens", "out_tokens", "dataset", "predicted_osl"])
    r = load_trace(path)[0]
    assert r.out_tokens == 500, r
    assert r.predicted_osl == 123, r


@check("test_a_trace_without_a_prediction_column_reads_zero_not_the_truth")
def _():
    path = write_trace([{"sample_id": 1, "in_tokens": 100, "out_tokens": 500,
                         "dataset": "cnn"}])
    assert load_trace(path)[0].predicted_osl == 0


@check("test_the_generator_hands_records_out_in_file_order")
def _():
    recs = [InferenceRecord(i, 10 * i, i, "d") for i in range(1, 5)]
    gen = make_trace_generator(recs)
    assert [gen().sample_id for _ in range(4)] == [1, 2, 3, 4]


@check("test_the_trace_wraps_and_the_wrap_is_predictable")
def _():
    # The documented mapping back to a sample:
    #     sample_id = (req_num - 1) mod len(records) + 1
    recs = [InferenceRecord(i, 10, 5, "d") for i in range(1, 4)]
    gen = make_trace_generator(recs)
    for req_num in range(1, 11):
        want = (req_num - 1) % len(recs) + 1
        assert gen().sample_id == want, req_num


@check("test_an_empty_trace_is_refused_rather_than_serving_nothing")
def _():
    try:
        make_trace_generator([])
    except ValueError:
        return
    raise AssertionError("an empty trace produced a generator")


@check("test_every_shipped_scenario_parses_and_is_non_empty")
def _():
    scen = ROOT / "loadgen" / "scenarios"
    files = sorted(scen.glob("*.csv"))
    assert files, "no scenarios found"
    for f in files:
        recs = load_trace(str(f))
        assert recs, f"{f.name} is empty"
        assert all(r.in_tokens > 0 for r in recs), f"{f.name} has a zero prompt"
        assert all(r.out_tokens > 0 for r in recs), f"{f.name} has a zero output"


# ─────────────────────────── one request's row ──────────────────────────────

REQUIRED = ("req_id", "sched_id", "dc_id", "status", "arrival_s", "send_s",
            "pool_wait_ms", "outbox_ms", "scheduling_ms", "forwarding_ms",
            "queue_ms",
            "ttft_ms", "avg_tpot_ms", "recv_s", "wall_ms", "latency_ms",
            "service_ms", "in_flight", "in_tok", "out_tok",
            "carbon_intensity", "send_time", "recv_time", "error")


@check("test_a_served_request_reports_what_the_scheduler_said")
def _():
    stub = FakeSchedStub(dc_id="US_MISO", ttft_ms=40.0, avg_tpot_ms=19.0,
                         queue_ms=3.0, scheduling_ms=0.5, forwarding_ms=120.0,
                         latency_ms=900.0, service_ms=880.0,
                         carbon_intensity=210.0)
    rec = InferenceRecord(1, 128, 64, "cnn", 0)
    row = send_one_request(stub, "sched-us_east", 7, rec)
    assert row["status"] == "OK", row
    assert row["req_id"] == "req7", row["req_id"]
    assert row["dc_id"] == "US_MISO", row
    assert row["ttft_ms"] == 40.0 and row["forwarding_ms"] == 120.0, row
    assert row["in_tok"] == 128 and row["out_tok"] == 64, row


@check("test_the_request_id_goes_on_the_wire")
def _():
    # It used to live only in the client's CSV while the scheduler minted its
    # own, so a row here and a row in DECISION_LOG could not be joined -- and
    # with the trace wrapping there was nothing else to join on.
    stub = FakeSchedStub()
    send_one_request(stub, "sched-us_east", 12, InferenceRecord(1, 10, 5, "d"))
    assert stub.seen[0].req_id == "req12", stub.seen[0].req_id


@check("test_the_dataset_and_prediction_are_carried_to_the_scheduler")
def _():
    stub = FakeSchedStub()
    send_one_request(stub, "sched-us_east", 1,
                     InferenceRecord(1, 10, 5, "govreport", 321))
    got = stub.seen[0]
    assert got.dataset == "govreport", got.dataset
    assert got.predicted_osl == 321, got.predicted_osl
    assert got.out_tok == 5, "ground truth was not sent"


@check("test_a_forwarded_request_reports_the_whole_path")
def _():
    stub = FakeSchedStub(sched_path="sched-us_east,sched-us_west")
    row = send_one_request(stub, "sched-us_east", 1,
                           InferenceRecord(1, 10, 5, "d"))
    assert row["sched_id"] == "sched-us_east,sched-us_west", row["sched_id"]


@check("test_a_failed_request_writes_the_same_columns_as_a_served_one")
def _():
    # Two column sets in one file is a broken CSV, and the failures are
    # exactly the rows an analysis needs to count.
    ok = send_one_request(FakeSchedStub(), "sched-us_east", 1,
                          InferenceRecord(1, 10, 5, "d"))
    bad = send_one_request(FakeSchedStub(raises=RuntimeError("refused")),
                           "sched-us_east", 2, InferenceRecord(2, 10, 5, "d"))
    assert list(ok) == list(bad), \
        f"different columns: {set(ok) ^ set(bad)}"
    for k in REQUIRED:
        assert k in ok, f"missing column {k}"
    assert bad["status"] == "ERROR" and bad["error"], bad
    assert ok["error"] == "" or ok["error"] is None or not ok["error"], ok["error"]


@check("test_pool_wait_measures_the_lag_and_not_the_loop")
def _():
    # arrival_s is where the schedule said the request belonged; send_s is when
    # it was actually issued. The gap is invisible in wall_ms, and it is the
    # signature of a generator that has stopped being open-loop. Under asyncio
    # the cause changed -- there is no pool to queue behind, so it is event-loop
    # lag -- but the column has to keep reporting it, or the next saturation is
    # silent the way the thread model's was.
    stub = FakeSchedStub()
    rec = InferenceRecord(1, 10, 5, "d")
    t0 = time.monotonic()
    time.sleep(0.05)
    row = send_one_request(stub, "s", 1, rec, t_run0=t0, arrival_s=0.0)
    assert row["pool_wait_ms"] >= 45.0, \
        f"a 50 ms late dispatch reported {row['pool_wait_ms']} ms of pool wait"

    row2 = send_one_request(stub, "s", 2, rec, t_run0=time.monotonic(),
                            arrival_s=0.0)
    assert row2["pool_wait_ms"] < 20.0, row2["pool_wait_ms"]


@check("test_pool_wait_is_never_negative")
def _():
    # A request issued EARLY (the schedule is in the future) must read 0, not a
    # negative wait that would cancel real waits out of a mean.
    row = send_one_request(FakeSchedStub(), "s", 1,
                           InferenceRecord(1, 10, 5, "d"),
                           t_run0=time.monotonic(), arrival_s=5.0)
    assert row["pool_wait_ms"] == 0.0, row["pool_wait_ms"]


@check("test_the_row_reads_as_the_request_life_in_order")
def _():
    row = send_one_request(FakeSchedStub(), "s", 1,
                           InferenceRecord(1, 10, 5, "d"),
                           t_run0=time.monotonic(), arrival_s=0.0)
    assert row["send_s"] <= row["recv_s"], (row["send_s"], row["recv_s"])
    assert row["wall_ms"] >= 0.0


# ───────────────────────────── the whole run ────────────────────────────────

import loadgen.client as client_mod                                # noqa: E402


def do_run(stubs, duration_s=1.0, rate=50.0, seed=7, n_records=20,
           concurrent=64, dist="poisson"):
    """Drive client.run against stubbed schedulers and read back its CSV.

    run() discovers its own peers and writes rather than returns, so the two
    discovery calls are replaced and the output file is the result.
    """
    recs = [InferenceRecord(i, 100 + i, 10, "cnn", 0)
            for i in range(1, n_records + 1)]
    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    real_disc = client_mod.discover_schedulers
    real_build = client_mod.build_scheduler_stubs
    client_mod.discover_schedulers = lambda: {k: "localhost:1" for k in stubs}
    class _NullChannel:
        async def close(self):
            pass
    client_mod.build_scheduler_stubs = lambda eps: (dict(stubs), [_NullChannel()])
    try:
        random.seed(seed)
        client_mod.run(
            duration_s=duration_s,
            arrival_gen=make_arrival_generator(dist, 1, 2, rate),
            token_gen=make_trace_generator(recs),
            seed=seed, output_csv=out, concurrent=concurrent, verbosity=0)
    finally:
        client_mod.discover_schedulers = real_disc
        client_mod.build_scheduler_stubs = real_build
    with open(out) as f:
        return list(csv.DictReader(f))


@check("test_a_run_serves_the_arrivals_and_reports_them_all")
def _():
    stub = FakeSchedStub(ttft_ms=10.0, latency_ms=20.0, service_ms=18.0)
    rows = do_run({"sched-us_east": stub}, duration_s=1.0, rate=50.0)
    assert len(rows) >= 20, f"only {len(rows)} arrivals in 1 s at 50/s"
    assert all(r["status"] == "OK" for r in rows), \
        [r for r in rows if r["status"] != "OK"][:2]
    assert len(stub.seen) == len(rows), (len(stub.seen), len(rows))


@check("test_a_run_is_reproducible_from_its_seed")
def _():
    def once():
        stub = FakeSchedStub()
        do_run({"sched-us_east": stub}, duration_s=0.6, rate=50.0, seed=99)
        return [(r.in_tok, r.out_tok, r.dataset) for r in stub.seen]
    a, b = once(), once()
    n = min(len(a), len(b))
    assert n > 5, n
    assert a[:n] == b[:n], "the same seed produced a different workload"


@check("test_requests_are_spread_over_every_scheduler")
def _():
    stubs = {f"sched-{g}": FakeSchedStub()
             for g in ("us_east", "us_west", "asia")}
    do_run(stubs, duration_s=1.0, rate=60.0, seed=5, n_records=10)
    counts = [len(s.seen) for s in stubs.values()]
    assert all(c > 0 for c in counts), f"a scheduler got nothing: {counts}"
    assert max(counts) - min(counts) <= 1, f"uneven entry points: {counts}"


@check("test_a_scheduler_that_refuses_does_not_stop_the_run")
def _():
    stub = FakeSchedStub(raises=RuntimeError("connection refused"))
    rows = do_run({"sched-us_east": stub}, duration_s=0.5, rate=40.0,
                  n_records=1)
    assert rows, "the run produced nothing"
    assert all(r["status"] == "ERROR" for r in rows)
    assert all(r["error"] for r in rows)


@check("test_the_output_csv_carries_every_column_the_row_has")
def _():
    rows = do_run({"sched-us_east": FakeSchedStub()}, duration_s=0.6,
                  rate=40.0, n_records=5)
    assert rows
    for k in REQUIRED:
        assert k in rows[0], f"missing column {k}"
    ids = [r["req_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate request ids in the output"


@check("test_the_achieved_rate_is_close_to_the_nominal_one")
def _():
    # The schedule is absolute, not sleep-for-the-interval, so per-iteration
    # overhead is absorbed rather than accumulated. If that regresses the
    # achieved rate drifts below nominal and the rate axis compresses.
    dur, rate = 2.0, 100.0
    rows = do_run({"sched-us_east": FakeSchedStub()}, duration_s=dur,
                  rate=rate, seed=11, n_records=1, concurrent=512)
    achieved = len(rows) / dur
    assert abs(achieved - rate) / rate < 0.10, \
        f"asked {rate} req/s, achieved {achieved:.1f}"


@check("test_an_explicit_cap_shows_up_as_wait_rather_than_hiding")
def _():
    # --concurrent > 0 makes a run deliberately closed-loop. That is a valid
    # thing to ask for, but it must be VISIBLE: the held requests have to
    # surface as pool_wait_ms rather than as a quietly lower offered rate.
    stub = FakeSchedStub(delay=0.30)
    rows = do_run({"sched-us_east": stub}, duration_s=1.0, rate=60.0,
                  n_records=5, concurrent=2)
    waits = [float(r["pool_wait_ms"]) for r in rows]
    assert max(waits) > 50.0, \
        f"a cap of 2 against a 300 ms scheduler showed no wait: {max(waits)}"


@check("test_a_slow_fleet_does_not_slow_the_arrival_loop")
def _():
    # THE property the rewrite exists for, and the one the thread model failed.
    # Uncapped, a scheduler that takes 300 ms per request must not delay the
    # NEXT arrival: requests are coroutines on one loop, so slow ones are
    # parked on I/O rather than holding a worker. Measured on our testbed, the thread
    # model failed exactly here -- 39789 threads blocked on gRPC starved the
    # arrival loop, and a 144001-slot pool at 28% occupancy still delivered
    # 716/s of a requested 1200 with pool_wait p50 at 5.2 s.
    stub = FakeSchedStub(delay=0.30)
    rows = do_run({"sched-us_east": stub}, duration_s=1.0, rate=60.0,
                  n_records=5, concurrent=0)
    waits = [float(r["pool_wait_ms"]) for r in rows]
    assert len(rows) >= 40, f"the loop did not keep its schedule: {len(rows)} rows"
    assert max(waits) < 100.0, \
        f"a 300 ms fleet delayed the arrivals by {max(waits)} ms"


# ──────────────────────────── the stream mux ────────────────────────────────
# One stream carries every request to a scheduler, so the reply has to be
# routed back by req_id. A routing fault does not raise: it puts one request's
# dc_id and service_ms on another request's row, in every row of the file. So
# the checks below are about identity, not throughput.


class _FakeStreamStub:
    """A scheduler that answers a stream, in an order of its choosing.

    Replies are emitted once `batch` of them have accumulated rather than at
    end of stream, because the client only half-closes after its requests are
    answered -- a stub that waited for the half-close would deadlock against
    it, which is also true of the real server and is why it does not.
    """

    def __init__(self, batch=1, order=None, fail_after=None, extra_id=None):
        self.batch = batch
        self.order = order            # callable: list of replies -> list
        self.fail_after = fail_after  # raise once this many are read
        self.extra_id = extra_id      # emit a reply nobody asked for
        self.seen = []

    def SubmitStream(self, request_iterator):
        async def _gen():
            buf = []
            async for req in request_iterator:
                self.seen.append(req.req_id)
                if self.fail_after is not None and \
                        len(self.seen) >= self.fail_after:
                    raise RuntimeError("stream broke")
                buf.append(inference_pb2.InferenceResponse(
                    req_id=req.req_id, dc_id=f"dc-for-{req.req_id}",
                    status="OK", service_ms=float(len(req.req_id))))
                if len(buf) >= self.batch:
                    for r in (self.order(buf) if self.order else buf):
                        yield r
                    buf = []
                    if self.extra_id is not None:
                        yield inference_pb2.InferenceResponse(
                            req_id=self.extra_id, dc_id="dc-ghost", status="OK")
                        self.extra_id = None
            for r in (self.order(buf) if self.order else buf):
                yield r
        return _gen()


def _mux(stub):
    from common.stream_mux import StreamMux
    return StreamMux(stub, "sched-test")


@check("test_stream_mux_routes_each_reply_to_its_own_request")
def _():
    # Replies come back exactly reversed, so every one of them is at the wrong
    # position. Only the req_id match can put them right.
    stub = _FakeStreamStub(batch=25, order=lambda b: list(reversed(b)))

    async def _run():
        m = _mux(stub)
        m.start()
        reqs = [inference_pb2.InferenceRequest(req_id=f"req{i}", in_tok=10,
                                               out_tok=5) for i in range(25)]
        got = await asyncio.wait_for(
            asyncio.gather(*[m.SubmitRequest(r) for r in reqs]), 10.0)
        await m.close()
        return reqs, got

    reqs, got = asyncio.run(_run())
    for req, resp in zip(reqs, got):
        assert resp.req_id == req.req_id, \
            f"{req.req_id} was answered with {resp.req_id}"
        assert resp.dc_id == f"dc-for-{req.req_id}", \
            f"{req.req_id} got {resp.dc_id}"


@check("test_stream_mux_drops_a_reply_nobody_is_waiting_for")
def _():
    # A duplicated or misrouted reply must not be handed to whichever request
    # happens to be waiting. It is dropped, and the run is told.
    stub = _FakeStreamStub(batch=4, extra_id="req-that-never-was")

    async def _run():
        m = _mux(stub)
        m.start()
        reqs = [inference_pb2.InferenceRequest(req_id=f"req{i}", in_tok=10,
                                               out_tok=5) for i in range(4)]
        got = await asyncio.wait_for(
            asyncio.gather(*[m.SubmitRequest(r) for r in reqs]), 10.0)
        await m.close()
        return got

    got = asyncio.run(_run())
    assert len(got) == 4, len(got)
    assert all(r.dc_id != "dc-ghost" for r in got), \
        "the unmatched reply was delivered to a real request"


@check("test_stream_mux_fails_its_waiters_when_the_stream_breaks")
def _():
    # Otherwise every in-flight request hangs until the run's drain timeout and
    # a dead stream reads as a stalled fleet.
    stub = _FakeStreamStub(batch=100, fail_after=3)

    async def _run():
        m = _mux(stub)
        m.start()
        reqs = [inference_pb2.InferenceRequest(req_id=f"req{i}", in_tok=10,
                                               out_tok=5) for i in range(10)]
        out = await asyncio.wait_for(
            asyncio.gather(*[m.SubmitRequest(r) for r in reqs],
                           return_exceptions=True), 10.0)
        await m.close()
        return out

    out = asyncio.run(_run())
    assert all(isinstance(o, Exception) for o in out), \
        f"a broken stream left requests waiting: {[type(o).__name__ for o in out]}"


@check("test_stream_mux_failure_becomes_an_error_row_not_an_escape")
def _():
    # The unary transport has test_a_scheduler_that_refuses_does_not_stop_the
    # _run; this is its streamed twin, and it exists because the two raise
    # different types. send_one_request catches Exception, so StreamClosed and
    # asyncio.TimeoutError both land as rows -- but if that clause were ever
    # narrowed to grpc.RpcError, a streamed failure would escape and arm A and
    # arm B of a transport comparison would disagree on how many requests the
    # run even had.
    from common.stream_mux import StreamMux

    stub = _FakeStreamStub(batch=100, fail_after=1)

    # send_one_request is wrapped for synchronous use at the top of this file;
    # drive the mux on the same loop it was started on instead.
    async def _drive():
        m = StreamMux(stub, "sched-us_east")
        m.start()
        row = await _send_one(m, "sched-us_east", 1,
                              InferenceRecord(1, 10, 5, "x"))
        await m.close()
        return row

    row = asyncio.run(_drive())
    assert row["status"] == "ERROR", row["status"]
    assert row["error"], "the failure produced no error text"
    for k in REQUIRED:
        assert k in row, f"the error row is missing {k}"


@check("test_stream_mux_reports_what_a_request_waited_in_the_outbox")
def _():
    # A queue with no instrument is how the admit gap hid for a week. This one
    # is the mux's outbox; the column must exist and be real, not always zero.
    from common.stream_mux import StreamMux

    class _SlowReader(_FakeStreamStub):
        def SubmitStream(self, request_iterator):
            async def _gen():
                await asyncio.sleep(0.05)      # nothing is read for 50 ms
                async for req in request_iterator:
                    yield inference_pb2.InferenceResponse(
                        req_id=req.req_id, dc_id="dc", status="OK")
            return _gen()

    async def _drive():
        m = StreamMux(_SlowReader(), "sched-us_east")
        m.start()
        row = await _send_one(m, "sched-us_east", 1,
                              InferenceRecord(1, 10, 5, "x"))
        await m.close()
        return row

    row = asyncio.run(_drive())
    assert row["status"] == "OK", row.get("error")
    assert row["outbox_ms"] >= 40.0, \
        f"a 50 ms stall was reported as {row['outbox_ms']} ms in the outbox"


@check("test_stream_mux_reconnects_for_a_caller_that_asked_for_it")
def _():
    # The scheduler's peer hop, not the generator's. The unary path recovers
    # through the channel on its own, so a mux that stayed broken would turn a
    # blip into a quarter of the fleet unable to forward for the rest of a run.
    from common.stream_mux import StreamMux

    class _FlakyStub:
        def __init__(self):
            self.streams = 0

        def SubmitStream(self, request_iterator):
            self.streams += 1
            first = self.streams == 1

            async def _gen():
                async for req in request_iterator:
                    if first:
                        raise RuntimeError("peer went away")
                    yield inference_pb2.InferenceResponse(
                        req_id=req.req_id, dc_id="dc", status="OK")
            return _gen()

    stub = _FlakyStub()

    async def _drive():
        m = StreamMux(stub, "sched-peer", reconnect=True)
        m.start()
        r = inference_pb2.InferenceRequest(req_id="a", in_tok=1, out_tok=1)
        try:
            await asyncio.wait_for(m.SubmitRequest(r), 5.0)
            broke = False
        except Exception:
            broke = True
        # The next request must get a NEW stream rather than the dead one.
        r2 = inference_pb2.InferenceRequest(req_id="b", in_tok=1, out_tok=1)
        ok = await asyncio.wait_for(m.SubmitRequest(r2), 5.0)
        await m.close()
        return broke, ok, stub.streams

    broke, ok, streams = asyncio.run(_drive())
    assert broke, "the first request should have failed with the stream"
    assert ok.req_id == "b", ok.req_id
    assert streams == 2, f"expected a second stream, got {streams}"


@check("test_stream_mux_stays_broken_for_a_caller_that_did_not")
def _():
    # The generator's contract: a broken stream means the run is void, and
    # silently re-establishing it would hide that.
    from common.stream_mux import StreamMux

    stub = _FakeStreamStub(batch=100, fail_after=1)

    async def _drive():
        m = StreamMux(stub, "sched-test")          # reconnect defaults to False
        m.start()
        out = []
        for i in range(2):
            try:
                await asyncio.wait_for(m.SubmitRequest(
                    inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=1,
                                                   out_tok=1)), 5.0)
                out.append("ok")
            except Exception:
                out.append("fail")
        await m.close()
        return out

    assert asyncio.run(_drive()) == ["fail", "fail"], \
        "a mux without reconnect answered after its stream broke"


@check("test_stream_mux_refuses_a_request_with_no_id")
def _():
    # The unary path lets the scheduler mint an id. Here the id is the only
    # thing the reply can be matched on, so a missing one is unroutable.
    stub = _FakeStreamStub()

    async def _run():
        m = _mux(stub)
        m.start()
        try:
            await m.SubmitRequest(inference_pb2.InferenceRequest(in_tok=1,
                                                                 out_tok=1))
        except ValueError:
            return True
        finally:
            await m.close()
        return False

    assert asyncio.run(_run()), "a request with no req_id was accepted"


if __name__ == "__main__":
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        sys.exit(1)

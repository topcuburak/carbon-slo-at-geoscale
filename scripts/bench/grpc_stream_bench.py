#!/usr/bin/env python3
"""How many requests a second can one grpc.aio server process actually retire?

The scheduler's event loop is 90% grpc/protobuf runtime -- parse, serialise,
one asyncio task per RPC -- and 10% our code, measured at 4400 req/s. So the
ceiling is a property of how the transport is used, and the question this
answers is whether ONE bidirectional stream per client, instead of one RPC per
request, moves it enough to be worth rewriting three hops for.

The handler does nothing but build a reply. That is deliberate: this measures
the floor cost of moving a request through grpc.aio, which is what "rest" is.

Shape matches production: N client processes against ONE server process, each
client keeping many requests in flight, because a scheduler is fed by several
load-generator shards and several peers.

  python3 scripts/bench/grpc_stream_bench.py            # both modes
  python3 scripts/bench/grpc_stream_bench.py --mode stream --clients 4

Read the RPS, and read `server cpu` beside it: if the server is under 100% of
a core the client side is the limit and the number is a floor, not a ceiling.
"""

import argparse
import asyncio
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grpc                                                   # noqa: E402
import bench_pb2                                              # noqa: E402
import bench_pb2_grpc                                         # noqa: E402

ADDR = "127.0.0.1:50999"


def _reply(req):
    return bench_pb2.Resp(
        req_id=req.req_id, dc_id="US_PJM", status="OK",
        service_ms=1234.5, latency_ms=1300.0, carbon_intensity=370.3,
        rtt_overhead_ms=0.0, sched_path="sched-us_east", ttft_ms=90.1,
        avg_tpot_ms=22.5, queue_ms=0.0, forwarding_ms=0.0,
        scheduling_ms=0.4, prefill_bs=12,
    )


class Servicer(bench_pb2_grpc.BenchServicer):
    async def Unary(self, request, context):
        return _reply(request)

    async def Stream(self, request_iterator, context):
        """Read requests, hand each to its own task, write replies as they land.

        The replies must NOT be written in arrival order: a stream that answers
        in order couples the latencies of everything behind a slow request,
        which is exactly the measurement the whole experiment is about. So the
        reader and the writer are separate and a queue joins them.
        """
        out = asyncio.Queue()
        inflight = 0
        reading = True

        async def _run(req):
            nonlocal inflight
            await out.put(_reply(req))
            inflight -= 1

        async def _read():
            nonlocal reading, inflight
            async for req in request_iterator:
                inflight += 1
                asyncio.create_task(_run(req))
            reading = False
            await out.put(None)         # wake the writer on a quiet stream

        reader = asyncio.create_task(_read())
        try:
            while reading or inflight > 0 or not out.empty():
                item = await out.get()
                if item is not None:
                    yield item
        finally:
            reader.cancel()


async def _serve(ready):
    server = grpc.aio.server()
    bench_pb2_grpc.add_BenchServicer_to_server(Servicer(), server)
    server.add_insecure_port(ADDR)
    await server.start()
    ready.set()
    await server.wait_for_termination()


def server_proc(ready):
    async def main():
        ev = asyncio.Event()
        task = asyncio.create_task(_serve(ev))
        await ev.wait()
        ready.set()
        await task
    asyncio.run(main())


def _req(i):
    return bench_pb2.Req(in_tok=841, out_tok=71, req_id=f"req{i}",
                         dataset="mlperf", predicted_osl=71,
                         user_id="", forwarded=False, target_dc="US_PJM")


async def _client_unary(seconds, concurrency, counter):
    async with grpc.aio.insecure_channel(ADDR) as ch:
        stub = bench_pb2_grpc.BenchStub(ch)
        stop = time.monotonic() + seconds
        n = [0]

        async def worker(base):
            i = base
            while time.monotonic() < stop:
                await stub.Unary(_req(i))
                n[0] += 1
                i += concurrency

        await asyncio.gather(*[worker(k) for k in range(concurrency)])
        counter.value += n[0]


async def _client_stream(seconds, concurrency, counter):
    async with grpc.aio.insecure_channel(ADDR) as ch:
        stub = bench_pb2_grpc.BenchStub(ch)
        stop = time.monotonic() + seconds
        sent = [0]
        got = [0]
        done = asyncio.Event()

        async def gen():
            i = 0
            while time.monotonic() < stop:
                # Keep `concurrency` messages outstanding, the same shape the
                # unary arm runs at, or the two are not comparable.
                while sent[0] - got[0] >= concurrency:
                    await asyncio.sleep(0)
                yield _req(i)
                sent[0] += 1
                i += 1
            done.set()

        call = stub.Stream(gen())
        async for _ in call:
            got[0] += 1
            if done.is_set() and got[0] >= sent[0]:
                break
        counter.value += got[0]


def client_proc(mode, seconds, concurrency, counter):
    fn = _client_unary if mode == "unary" else _client_stream
    asyncio.run(fn(seconds, concurrency, counter))


def _cpu_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        return (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return 0.0


def run(mode, clients, seconds, concurrency):
    ready = mp.Event()
    srv = mp.Process(target=server_proc, args=(ready,), daemon=True)
    srv.start()
    if not ready.wait(20):
        print("server did not start")
        srv.terminate()
        return

    counter = mp.Value("l", 0)
    c0 = _cpu_of(srv.pid)
    t0 = time.monotonic()
    procs = [mp.Process(target=client_proc,
                        args=(mode, seconds, concurrency, counter))
             for _ in range(clients)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.monotonic() - t0
    cpu = _cpu_of(srv.pid) - c0
    srv.terminate()
    srv.join(5)

    rps = counter.value / wall if wall > 0 else 0.0
    print(f"  {mode:<7} clients={clients} inflight/client={concurrency}  "
          f"{rps:9.0f} req/s   server cpu={100.0 * cpu / wall:5.1f}%   "
          f"{1e6 * cpu / max(1, counter.value):6.1f} us cpu/req")
    return rps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unary", "stream", "both"],
                    default="both")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--inflight", type=int, default=256)
    a = ap.parse_args()

    print(f"grpc {grpc.__version__}  python {sys.version.split()[0]}  "
          f"{a.seconds:.0f}s per arm")
    modes = ["unary", "stream"] if a.mode == "both" else [a.mode]
    got = {}
    for m in modes:
        got[m] = run(m, a.clients, a.seconds, a.inflight)
    if len(got) == 2 and got["unary"]:
        print(f"\n  stream / unary = {got['stream'] / got['unary']:.2f}x")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

"""The within-DC and within-node schedulers.

There are three placement decisions in this simulator, and they are different
problems with different information:

    1. GLOBAL       request    -> datacentre   scheduler/policies/. Carbon and
                    SLA aware, deciding on a polled snapshot that is an epoch
                    stale and reached over a network.
    2. WITHIN-DC    datacentre -> node         this file, run by Datacenter.
                    Local and synchronous, and the node is a power domain.
    3. WITHIN-NODE  node       -> engine       this file, run by Node. Local,
                    and the engine is what actually holds the batch and the KV.

Levels 2 and 3 used to be one. A single flat round-robin ran over every engine
in the datacentre on one counter under one lock, so the node boundary played no
part in scheduling and `Node` carried no state beyond its own id. Splitting them gives each level its
own policy, its own counter and its own lock, and gives the node a place to
stand -- it is the domain that shares host power, memory bandwidth and a PCIe
root. Those turn out to cost almost nothing: one TP=2 engine at 8 req/s reads
TPOT p50 29.45 ms against 30.14 ms for two at the same per-engine rate, and
35.65 against 35.68 at p90. The node earns its place for placement and
accounting, not for contention.

Adding a policy means writing a class with a `pick` and decorating it with
`@register`; neither scheduler needs to change.

    from datacenter.scheduling import make_scheduler
    sched = make_scheduler("kv_headroom")     # at either level
    chosen = sched.pick(candidates)

A policy declares the metric it keys on and reads it off the candidate itself,
so the caller does not have to know what a given policy looks at. Nodes and
engines name their metrics identically -- `load`, `kv_utilisation` -- which is
what lets the same class place at either level.

A policy instance owns mutable state and is NOT shared between the two levels
or between nodes: every node builds its own within-node scheduler, which is
what makes the nodes independent of one another.
"""

from __future__ import annotations

import operator
import threading
from typing import Callable, Dict, List, Sequence, Type

__all__ = ["SchedulingPolicy", "RoundRobin", "LeastLoaded", "KvHeadroom",
           "make_scheduler", "register", "available_policies"]


class SchedulingPolicy:
    """Chooses one candidate from a fixed, ordered list.

    A policy declares the `metric` it keys on -- an attribute read off a
    candidate, lower being preferred -- and the caller does not need to know
    what that is. Nodes and inference engines expose the same metric names
    (`load`, `kv_utilisation`), so one class places at either level without
    being told which. Adding a policy that keys on something new means adding
    that property to both and nothing else.

    `pick` is called from many request threads at once and must be safe under
    that.
    """

    name: str = ""
    metric: str = ""

    def __init__(self, load_fn: Callable[[object], float] | None = None) -> None:
        if load_fn is None and self.metric:
            load_fn = operator.attrgetter(self.metric)
        self.load_fn = load_fn

    def pick(self, candidates: Sequence):
        raise NotImplementedError


_REGISTRY: Dict[str, Type[SchedulingPolicy]] = {}


def register(cls: Type[SchedulingPolicy]) -> Type[SchedulingPolicy]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name to be registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_policies() -> List[str]:
    return sorted(_REGISTRY)


def make_scheduler(name: str, load_fn: Callable[[object], float] | None = None
                ) -> SchedulingPolicy:
    """Build one policy instance. Each caller gets its own state.

    `load_fn` overrides the class's declared metric; leave it out and the
    policy reads its own.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown placement policy {name!r}; have {available_policies()}"
        ) from None
    return cls(load_fn=load_fn)


@register
class RoundRobin(SchedulingPolicy):
    """Cyclic, load-blind.

    Balances the request COUNT exactly and the WORK not at all: measured over
    a mixed trace across four engines it gave 150/150/150/150 requests but a
    21% spread in prompt tokens, and an instantaneous batch gap between engines
    of 16 at p90 on a mean of 12.7. That spread does not reach the step models,
    which are near-linear over the range the simulator visits, so this remains
    the default -- it is the behaviour every result so far was produced under.
    """

    name = "round_robin"

    def __init__(self, load_fn=None) -> None:
        super().__init__(load_fn)
        self._i = 0
        self._lock = threading.Lock()

    def pick(self, candidates: Sequence):
        with self._lock:
            k = self._i % len(candidates)
            self._i += 1
        return candidates[k]


@register
class LeastLoaded(SchedulingPolicy):
    """Fewest in flight, with the ties rotated.

    The rotation is the point. A plain `min` over equal loads returns the first
    candidate every time, so an idle datacentre sends every arrival to engine 0
    until its count rises -- the herd, and the reason the least-loaded rungs
    were taken out of the datacentre-level policies. Rotating the tie-break
    spreads a tied field exactly as round-robin would.

    What it still cannot see: picks already made whose request has not yet
    reached the engine. That window is not the epoch-stale one that made the
    herd bad at level 1 -- here `process` raises `pending_count` synchronously
    on the calling thread, microseconds later -- but it is not zero, and under
    enough concurrent dispatch several picks can land before any of them shows
    up. Whether that matters is a measurement, not an argument.
    """

    name = "least_loaded"
    metric = "load"

    def __init__(self, load_fn: Callable[[object], float] | None = None) -> None:
        super().__init__(load_fn)
        if self.load_fn is None:
            raise ValueError(f"{self.name} has no metric and no load_fn")
        self._rotor = 0
        self._lock = threading.Lock()

    def pick(self, candidates: Sequence):
        n = len(candidates)
        with self._lock:
            start = self._rotor % n
            self._rotor += 1
        best, best_load = None, None
        # Start the scan at a different candidate each call, so a tie resolves
        # to a different one each time instead of always the first.
        for off in range(n):
            c = candidates[(start + off) % n]
            load = self.load_fn(c)
            if best_load is None or load < best_load:
                best, best_load = c, load
        return best


@register
class KvHeadroom(LeastLoaded):
    """Most free KV cache, ties rotated.

    Same search as least_loaded against a different quantity, and the two do
    not agree: a seat and a block are different resources. A request holding a
    long prompt occupies one seat and a large multiple of the cache a short one
    does, so an engine can be light on requests and heavy on memory. vLLM
    admits on both -- max_num_seqs and free blocks -- and placing on request
    count alone is blind to the second.

    On this testbed only the first has ever bound: the cache runs 15-33% full
    and `kv_blocked_steps` stays at zero. This exists so that stays a
    measurement. It is the policy to reach for when contexts get long enough
    that it does not.
    """

    name = "kv_headroom"
    metric = "kv_utilisation"

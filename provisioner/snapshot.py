"""Getting the snapshot to every scheduler, and deciding who is in it.

A snapshot is what a scheduler knows about the fleet. Two ways of delivering
it, and the difference is the point rather than a tuning knob:

  sync    one identical snapshot to every scheduler, every period. They all
          see the same state at the same instant, so they make the same
          decision at the same instant -- they herd.
  async   the fleet is split into waves and each wave is pushed as it is
          polled. Schedulers still end up with a complete view, because they
          MERGE rather than replace, but they reach it at staggered times and
          therefore stop seeing identical state. Breaks the symmetry.

Wave membership is fixed across cycles so a site's refresh period is
predictable at `wave_count * wave_interval_s`.
"""

import logging
from concurrent import futures
from typing import Dict, List

import inference_pb2

log = logging.getLogger("provisioner.snapshot")

#: How long the push fan-out may take before the stragglers are given up on.
PUSH_DEADLINE_S = 5.0


class SnapshotPusher:
    """Sends snapshots to schedulers and splits the fleet into waves."""

    def __init__(self, peers, wave_count: int = 4) -> None:
        self.peers = peers
        self.wave_count = max(1, int(wave_count))

    def push_one(self, sid: str, snapshot: inference_pb2.Snapshot) -> None:
        self.peers.sched_stub(sid).PushSnapshot(snapshot, timeout=PUSH_DEADLINE_S)

    def push(self, epoch: int, dc_view: Dict[str, inference_pb2.DcStats],
             rtt_matrix: Dict[str, float]) -> None:
        """Push one snapshot, full or partial, to every scheduler.

        `dc_view` is whatever was freshly polled -- the whole fleet in sync
        mode, one wave in async. The RTT matrix is computed from the full
        cached fleet by the caller, so a wave never skews congestion.
        """
        snapshot = inference_pb2.Snapshot(
            epoch=epoch, dc_view=dc_view, rtt_matrix=rtt_matrix,
        )
        # Same two corrections as the poll fan-out: the deadline is caught, and
        # the pool is not a context manager so a slow scheduler cannot hold the
        # cycle open for as long as it takes to answer.
        pool = futures.ThreadPoolExecutor(max_workers=100)
        try:
            future_map = {
                pool.submit(self.push_one, sid, snapshot): sid
                for sid in self.peers.sched_endpoints
            }
            try:
                for fut in futures.as_completed(future_map, timeout=PUSH_DEADLINE_S):
                    sid = future_map[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        log.warning("Push to scheduler %s failed: %s", sid, e)
            except futures.TimeoutError:
                late = sorted(x for f, x in future_map.items() if not f.done())
                log.warning(
                    "Push deadline passed with %d of %d scheduler(s) still "
                    "unacknowledged (%s%s). They will get the next snapshot.",
                    len(late), len(future_map), ", ".join(late[:5]),
                    " ..." if len(late) > 5 else "")
        finally:
            pool.shutdown(wait=False)

    def waves(self, dc_ids):
        """Deterministic split of DC ids into ``wave_count`` waves.

        Sorted + round-robin so each DC lands in a fixed wave across
        cycles. Stable wave membership makes per-DC refresh period
        predictable (= ``wave_count * wave_interval_s``).
        """
        dc_ids_sorted = sorted(dc_ids)
        waves = [[] for _ in range(self.wave_count)]
        for idx, dc_id in enumerate(dc_ids_sorted):
            waves[idx % self.wave_count].append(dc_id)
        return waves

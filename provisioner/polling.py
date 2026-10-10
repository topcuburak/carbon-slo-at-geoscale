"""Asking every datacentre what it is doing, and remembering the answer.

Two jobs that belong together because the second one exists to cover for the
first. The fan-out is deadline-bounded, so a site that does not answer in time
is simply absent from that epoch's view -- and the cache is what stands in for
it, which is also what makes async waves work: only a quarter of the fleet is
polled per push, but the congestion model still needs all of it.

The cache therefore has to forget as well as remember. See `remember`.
"""

import csv
import logging
import os
import time
from concurrent import futures
from typing import Dict

import inference_pb2

log = logging.getLogger("provisioner.polling")

#: How long the whole fan-out may take before the stragglers are given up on.
POLL_DEADLINE_S = 5.0


class FleetPoller:
    """Polls datacentres and keeps the last known view of every one of them."""

    def __init__(self, peers, stale_after_epochs: int) -> None:
        self.peers = peers
        self.epoch = 0
        self.last_poll_duration = 0.0
        self.stale_after_epochs = stale_after_epochs
        # Persistent per-DC state cache. In async mode each wave only refreshes
        # a subset, but the congestion model always uses the full cached fleet
        # view so it is never misled by partial updates.
        self.view: Dict[str, inference_pb2.DcStats] = {}
        # Per-epoch fleet LOAD series. Energy is deliberately not in it: the
        # engines integrate power over a window whose length is only known
        # once the run ends (meta_node.energy_stats, "zero until compute_energy
        # has run"), so a poll reads 0 and always would. Recording four columns
        # of zeros is worse than recording none -- someone differences them and
        # gets nothing, silently.
        #
        # What IS here is the half that cannot be reconstructed afterwards:
        # when the fleet reached steady state and when it began draining.
        # Warm-up and drain are not the operating point being measured and are
        # a different size for every policy -- at 1500 req/s the drain was 20%
        # of a round_robin run and 55% of a carbon_aware one -- so any windowed
        # figure has to be drawn at a boundary this series identifies.
        # Windowing the ENERGY as well needs the engines to integrate
        # incrementally, which is an energy-model change, not plumbing.
        self._ts_path = os.environ.get("PROVISIONER_TIMESERIES", "")
        self._ts_writer = None
        self._seen_epoch: Dict[str, int] = {}

    def poll_one(self, dc_id: str, epoch: int) -> inference_pb2.PollStatsResponse:
        stub = self.peers.dc_stub(dc_id)
        return stub.PollStats(
            inference_pb2.PollStatsRequest(src="provisioner", epoch=epoch),
            timeout=5.0,
        )

    def poll(self, dc_ids) -> Dict[str, inference_pb2.DcStats]:
        """Fan-out poll to a specific subset of DCs concurrently.

        Accepts any iterable of dc_ids; used for both full-fleet (sync
        mode) and partial wave (async mode) polling.
        """
        self.epoch += 1
        dc_view: Dict[str, inference_pb2.DcStats] = {}

        dc_ids = list(dc_ids)
        if not dc_ids:
            return dc_view

        # NOT a context manager, and the timeout is caught. Both matter, and
        # both were measured: with `with ... as pool` the block waits for every
        # future on the way out, so one slow site stalls the cycle for as long
        # as it takes -- which is exactly what the 5 s deadline exists to
        # prevent. And `as_completed` RAISES when the deadline passes with
        # futures still pending; the try/except below sits inside the loop and
        # catches a failing RPC, not that. Uncaught, it left `run` -- which had
        # no handler either -- and killed the polling thread for good, while the
        # gRPC server went on answering GetStatus and every scheduler kept its
        # last snapshot forever. One slow datacentre or one slow scheduler was
        # enough; measured, the loop died after a single epoch.
        pool = futures.ThreadPoolExecutor(max_workers=100)
        try:
            future_map = {
                pool.submit(self.poll_one, dc_id, self.epoch): dc_id
                for dc_id in dc_ids
                if dc_id in self.peers.dc_endpoints
            }
            try:
                for fut in futures.as_completed(future_map, timeout=POLL_DEADLINE_S):
                    dc_id = future_map[fut]
                    try:
                        resp = fut.result()
                        dc_view[dc_id] = inference_pb2.DcStats(
                            active_requests=resp.active_requests,
                            pending_requests=resp.pending_requests,
                            num_gpus=resp.num_gpus,
                            carbon_intensity=resp.carbon_intensity,
                            total_pending_isl=resp.total_pending_isl,
                            total_prefill_isl=resp.total_prefill_isl,
                            prefilling_requests=resp.prefilling_requests,
                            total_active_isl=resp.total_active_isl,
                            total_generated_tokens=resp.total_generated_tokens,
                            decoding_requests=resp.decoding_requests,
                            total_remaining_osl=resp.total_remaining_osl,
                            prefill_duty=resp.prefill_duty,
                            admitted_requests=resp.admitted_requests,
                            kv_queue_ms=resp.kv_queue_ms,
                            kv_ms_per_token=resp.kv_ms_per_token,
                            kv_room_tokens=resp.kv_room_tokens,
                            pue=resp.pue,
                        )
                        self._record(dc_id, resp)
                    except Exception as e:
                        log.warning("Poll %s failed: %s", dc_id, e)
            except futures.TimeoutError:
                late = sorted(d for f, d in future_map.items() if not f.done())
                log.warning(
                    "Poll deadline passed with %d of %d datacentre(s) still "
                    "unanswered (%s%s). Continuing on the %d that replied; "
                    "the rest keep their cached view.",
                    len(late), len(future_map), ", ".join(late[:5]),
                    " ..." if len(late) > 5 else "", len(dc_view))
        finally:
            # Do not wait: a site that missed the deadline must not hold up the
            # next cycle. Its thread finishes on its own and its answer is
            # discarded.
            pool.shutdown(wait=False)

        return dc_view

    _TS_COLS = ("t_ms", "epoch", "dc_id", "active", "pending",
                "prefilling", "decoding", "prefill_duty", "num_gpus",
                "kv_tokens", "kv_capacity", "kv_blocked",
                "kv_queue_ms", "kv_ms_per_token", "kv_room_tokens")

    def _record(self, dc_id, resp) -> None:
        """One row per site per epoch: what that site was carrying."""
        if not self._ts_path:
            return
        if self._ts_writer is None:
            d = os.path.dirname(self._ts_path)
            if d:
                os.makedirs(d, exist_ok=True)
            # Truncate, do not append. One provisioner owns one run's series,
            # so appending only ever happens when a results directory is reused
            # -- and then the file holds two runs back to back, with epochs
            # repeating and the cumulative counters restarting midway. That is
            # silent: every column still parses. It cost a cell of seats90 and
            # the duty differencing read 0.95 off it before anyone noticed.
            f = open(self._ts_path, "w", newline="", buffering=1)
            self._ts_writer = csv.writer(f)
            self._ts_writer.writerow(self._TS_COLS)
        self._ts_writer.writerow([
            round(time.monotonic() * 1000.0, 3), self.epoch, dc_id,
            resp.active_requests, resp.pending_requests,
            resp.prefilling_requests, resp.decoding_requests,
            round(resp.prefill_duty, 4), resp.num_gpus,
            resp.kv_tokens, resp.kv_capacity_tokens, resp.kv_blocked_steps,
            round(resp.kv_queue_ms, 1), round(resp.kv_ms_per_token, 4),
            round(resp.kv_room_tokens),
        ])

    def poll_all(self) -> Dict[str, inference_pb2.DcStats]:
        """Fan-out poll to every registered DC."""
        return self.poll(self.peers.dc_endpoints.keys())

    def remember(self, dc_view: Dict[str, inference_pb2.DcStats]) -> None:
        """Fold a poll's answers into the cache, and drop what has gone.

        Two ways an entry stops being evidence. It can leave the fleet, in
        which case it is not a stale reading but a wrong one and goes at once.
        Or it can stay in discovery and stop answering, which is a blip for a
        cycle or two and a dead site after that -- the cache exists to ride out
        the former, so the latter needs a bound rather than an immediate drop.
        """
        self.view.update(dc_view)
        for dc_id in dc_view:
            self._seen_epoch[dc_id] = self.epoch

        gone = [d for d in self.view if d not in self.peers.dc_endpoints]
        stale = [d for d, e in self._seen_epoch.items()
                 if d in self.view and d not in gone
                 and self.epoch - e > self.stale_after_epochs]
        for d in gone + stale:
            self.view.pop(d, None)
            self._seen_epoch.pop(d, None)
        if gone:
            log.info("Dropped %d datacentre(s) that left discovery: %s",
                     len(gone), ", ".join(sorted(gone)[:5]))
        if stale:
            log.warning(
                "Dropped %d datacentre(s) that have not answered for %d "
                "epochs: %s. They no longer contribute to the congestion "
                "estimate.", len(stale), self.stale_after_epochs,
                ", ".join(sorted(stale)[:5]))

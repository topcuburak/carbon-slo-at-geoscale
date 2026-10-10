"""The cycle: poll the fleet, price congestion, push to every scheduler.

This is the whole provisioner in one object, and it does no work itself -- it
owns a `PeerRegistry`, a `FleetPoller`, a `CongestionModel` and a
`SnapshotPusher`, and decides when each of them runs. Everything substantive
lives in those four; what is here is the cadence and the guarantee that it
never stops.

That guarantee is the reason the loop body is a separate method. This thread is
the only thing that gives a scheduler its view of the fleet, and it is an
unsupervised daemon: when it used to die -- and one peer slower than the poll
deadline was enough -- the gRPC server kept answering and every scheduler
silently kept its last snapshot for the rest of the run. A cycle that fails is
now a cycle skipped, logged with a running count, and retried, and the cadence
holds either way.
"""

import logging
import threading
import time

from provisioner.congestion import CongestionModel
from provisioner.peers import PeerRegistry
from provisioner.polling import FleetPoller
from provisioner.snapshot import SnapshotPusher

log = logging.getLogger("provisioner")


class Provisioner:
    """Core polling + snapshot logic, runs in a background thread.

    Two modes, and the difference is the experiment rather than a preference:

      "sync"   poll the whole fleet, push one identical snapshot to every
               scheduler each period. They all see the same state, so they
               herd.
      "async"  split the fleet into `wave_count` waves and push each as it is
               polled. A scheduler still reaches a complete view -- it merges
               rather than replaces -- but at staggered times, so the
               schedulers stop seeing identical state. Breaks the symmetry.

    A mistyped mode raises. It used to fall through to "sync" without a word,
    which is the worst possible default: running sync while believing it was
    async makes the comparison meaningless and leaves no trace.
    """

    def __init__(self, poll_period_s: float, discovery_refresh_s: float,
                 rtt_beta: float = 0.5,
                 rtt_norm: str = "cap",
                 rtt_heat_cap: float = 32.0,
                 poll_mode: str = "sync",
                 wave_count: int = 4,
                 wave_interval_s: float = 0.5):
        if poll_mode not in ("sync", "async"):
            raise ValueError(
                f"poll_mode must be 'sync' or 'async', got {poll_mode!r}")
        self.poll_period_s = poll_period_s
        self.discovery_refresh_s = discovery_refresh_s
        self.poll_mode = poll_mode
        self.wave_count = max(1, int(wave_count))
        self.wave_interval_s = max(0.0, float(wave_interval_s))

        self.peers = PeerRegistry()
        # How many epochs a datacentre may go unanswered before its cached
        # reading is dropped. In async mode a site is only polled once per
        # wave_count epochs, so the bound has to clear a full cycle with room
        # for a couple of misses.
        self.poller = FleetPoller(
            self.peers, stale_after_epochs=max(3 * self.wave_count, 6))
        self.congestion = CongestionModel(
            beta=rtt_beta, norm=rtt_norm, heat_cap=rtt_heat_cap)
        self.pusher = SnapshotPusher(self.peers, wave_count=self.wave_count)

        self._last_discovery = 0.0
        self._stop = threading.Event()

    # -- what GetStatus reports, kept here so the servicer stays a shim --
    @property
    def epoch(self) -> int:
        return self.poller.epoch

    @property
    def last_poll_duration(self) -> float:
        return self.poller.last_poll_duration

    @property
    def dc_endpoints(self):
        return self.peers.dc_endpoints

    @property
    def sched_endpoints(self):
        return self.peers.sched_endpoints

    def run(self):
        """Main polling loop — runs until stop() is called."""
        self.peers.refresh()
        self._last_discovery = time.monotonic()

        consecutive_errors = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._one_cycle(t0)
                consecutive_errors = 0
            except BaseException:
                consecutive_errors += 1
                log.exception(
                    "Polling cycle failed (%d in a row). Schedulers are now "
                    "working from a snapshot that is at least one cycle stale.",
                    consecutive_errors)

            elapsed = time.monotonic() - t0
            if self.poll_mode == "sync":
                sleep_s = max(0.0, self.poll_period_s - elapsed)
            else:
                sleep_s = max(0.0, self.wave_count * self.wave_interval_s - elapsed)
            self._stop.wait(timeout=sleep_s)

    def _one_cycle(self, t0: float):
        """One poll-and-push cycle. Anything it raises is caught by `run`."""
        if t0 - self._last_discovery > self.discovery_refresh_s:
            self.peers.refresh()
            self._last_discovery = t0

        if not self.peers.dc_endpoints:
            log.warning("No datacenter endpoints — skipping poll")
        elif self.poll_mode == "sync":
            view = self.poller.poll_all()
            self.poller.remember(view)
            self.poller.last_poll_duration = time.monotonic() - t0
            log.info(
                "Epoch %d (sync): polled %d/%d DCs in %.1fms  -> pushing full "
                "snapshot to %d schedulers",
                self.poller.epoch, len(view), len(self.peers.dc_endpoints),
                self.poller.last_poll_duration * 1000,
                len(self.peers.sched_endpoints),
            )
            self._push(view)
        else:
            for idx, wave in enumerate(self.pusher.waves(self.peers.dc_endpoints)):
                if self._stop.is_set():
                    break
                wave_t0 = time.monotonic()
                partial = self.poller.poll(wave)
                self.poller.remember(partial)
                log.info(
                    "Epoch %d (async wave %d/%d): polled %d DCs in %.1fms  -> "
                    "pushing partial snapshot",
                    self.poller.epoch, idx + 1, self.wave_count, len(partial),
                    (time.monotonic() - wave_t0) * 1000.0,
                )
                self._push(partial)
                if idx < self.wave_count - 1:
                    slept = time.monotonic() - wave_t0
                    self._stop.wait(timeout=max(0.0, self.wave_interval_s - slept))
            self.poller.last_poll_duration = time.monotonic() - t0

    def _push(self, view):
        """Push what was just polled, priced against the WHOLE cached fleet."""
        rtt = self.congestion.rtt_matrix(self.poller.view)
        self.pusher.push(self.poller.epoch, view, rtt)

    def stop(self):
        self._stop.set()

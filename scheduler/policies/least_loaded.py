"""Global least-loaded policy with cross-group forwarding support.

Picks the datacenter with the fewest ``active + pending`` requests
across ALL scheduler groups. Including ``pending_requests`` is critical:
a DC with ``active=5, pending=100`` is really in flight with 105 items
even though its GPU semaphore only holds 5 of them at a time, and
ranking it as "lightly loaded" was the root cause of the
``least_loaded`` herd-behavior pathology at high arrival rates.

Optionally rejects DCs whose ``active + pending`` already exceeds
``bs_cap`` to harden against stale-snapshot / cross-scheduler herds.
"""

from typing import Dict, Optional, Tuple

import inference_pb2
from scheduler.policies.common.capacity import at_cap
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope


@register_policy("least_loaded")
class LeastLoaded(BasePolicy):
    """Pick the DC with the fewest (active + pending) requests."""

    def __init__(self, bs_cap: int = 0, **kwargs):
        super().__init__(**kwargs)
        # Hard per-DC cap on (active + pending). 0 disables.
        self.bs_cap = bs_cap

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.GLOBAL

    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        result = self.choose_dc_extended(dc_view, region_id, **kwargs)
        return result[0] if result else None

    def choose_dc_extended(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[Tuple[str, bool]]:
        best_dc = None
        best_load = float("inf")
        fallback_dc = None
        fallback_load = float("inf")
        for dc_id, stats in dc_view.items():
            active = int(getattr(stats, 'active_requests', 0))
            pending = int(getattr(stats, 'pending_requests', 0))
            total_load = active + pending
            # Track an always-valid fallback ignoring the cap so we never
            # return None just because every DC is momentarily over bs_cap.
            if total_load < fallback_load:
                fallback_load = total_load
                fallback_dc = dc_id
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue
            if total_load < best_load:
                best_load = total_load
                best_dc = dc_id
        chosen = best_dc if best_dc is not None else fallback_dc
        if chosen is None:
            return None
        return (chosen, chosen in self._local_regions)

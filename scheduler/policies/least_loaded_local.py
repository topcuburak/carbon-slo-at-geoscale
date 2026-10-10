"""Least-loaded policy restricted to the local scheduler group.

Same logic as ``LeastLoaded`` (global) but only considers datacenters
in this scheduler's own geographic group — never forwards cross-group.
Uses ``active + pending`` as the load metric and supports an optional
hard ``bs_cap`` per DC for herd-hardening.
"""

from typing import Dict, Optional

import inference_pb2
from scheduler.policies.common.capacity import at_cap
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope


@register_policy("least_loaded_local")
class LeastLoadedLocal(BasePolicy):
    """Pick the local DC with the fewest (active + pending) requests."""

    def __init__(self, bs_cap: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.bs_cap = bs_cap

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.LOCAL_ONLY

    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        best_dc = None
        best_load = float("inf")
        fallback_dc = None
        fallback_load = float("inf")
        for dc_id, stats in dc_view.items():
            if dc_id not in self._local_regions:
                continue
            active = int(getattr(stats, 'active_requests', 0))
            pending = int(getattr(stats, 'pending_requests', 0))
            total_load = active + pending
            if total_load < fallback_load:
                fallback_load = total_load
                fallback_dc = dc_id
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue
            if total_load < best_load:
                best_load = total_load
                best_dc = dc_id
        return best_dc if best_dc is not None else fallback_dc

"""Probabilistic least-loaded policy.

DCs with fewer active requests get higher selection probability,
avoiding the herd-effect that can occur with deterministic
least-loaded (where every scheduler picks the same DC).
"""

import random
from typing import Dict, Optional

import inference_pb2
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope


@register_policy("weighted_least_loaded")
class WeightedLeastLoaded(BasePolicy):
    """Probabilistic selection inversely weighted by active requests."""

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.LOCAL_ONLY

    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        if not dc_view:
            return None
        ids = list(dc_view.keys())
        # active + pending, not active alone. A datacentre with active=5 and
        # pending=100 is carrying 105; its GPU semaphore only admits 5 at a
        # time. least_loaded's docstring records that reading active alone was
        # the root cause of its herd pathology -- and this policy, written to
        # avoid herding, was still reading it that way.
        loads = [int(getattr(dc_view[d], "active_requests", 0))
                 + int(getattr(dc_view[d], "pending_requests", 0))
                 for d in ids]
        max_load = max(loads) + 1  # +1 to avoid zero weights
        weights = [max_load - l for l in loads]
        return random.choices(ids, weights=weights, k=1)[0]

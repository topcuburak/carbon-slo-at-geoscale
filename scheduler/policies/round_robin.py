"""Global round-robin policy with cross-group forwarding support.

Cycles through ALL known datacenters in sorted order, regardless of
which scheduler group they belong to.
"""

from typing import Dict, Optional, Tuple

import inference_pb2
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope


@register_policy("round_robin")
class RoundRobin(BasePolicy):
    """Round-robin across all known datacenters (global)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx = 0

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
        ids = sorted(dc_view.keys())
        if not ids:
            return None
        dc = ids[self._idx % len(ids)]
        self._idx += 1
        return (dc, dc in self._local_regions)

"""Round-robin policy restricted to the local scheduler group.

Cycles through datacenters in this scheduler's own geographic group
only — never forwards cross-group.
"""

from typing import Dict, Optional

import inference_pb2
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope


@register_policy("round_robin_local")
class RoundRobinLocal(BasePolicy):
    """Round-robin across local group datacenters only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx = 0

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.LOCAL_ONLY

    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        ids = sorted(dc_id for dc_id in dc_view if dc_id in self._local_regions)
        if not ids:
            return None
        dc = ids[self._idx % len(ids)]
        self._idx += 1
        return dc

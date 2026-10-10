"""Base class and scope enum for scheduler routing policies.

Every policy must inherit from ``BasePolicy`` and declare a ``scope``:

- ``LOCAL_ONLY``  — choose_dc() picks from the local group; no forwarding.
- ``GLOBAL``      — choose_dc_extended() returns (dc_id, is_local) so the
                    scheduler can forward to a remote group when beneficial.
- ``GLOBAL_WITH_LOCAL`` — like GLOBAL, but also overrides
                    choose_dc_local_only() for handling forwarded requests.

There is a fifth method, and it is deliberately NOT defined here:
``choose_dc_batch``. A policy that wants requests handed to it in batches
rather than one at a time implements it, and the scheduler detects that with
``hasattr(policy, "choose_dc_batch")``. Defining it on this class -- even as a
stub that raises -- would make every policy look like a batch policy and route
all of them through the buffer. Its signature, from the two ILP policies::

    def choose_dc_batch(
        self,
        requests: List[RequestInfo],          # from common.estimators
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        max_batch_size: int = 0,
        rtt_matrix: Dict[str, float] = None,
    ) -> Dict[str, str]:                      # req_id -> dc_id

A batch policy still needs ``choose_dc`` and ``choose_dc_extended``: a forwarded
request arrives outside any batch and has to be answered on its own.

Quick-start for writing a custom policy::

    from scheduler.policies import register_policy
    from scheduler.policies.common.base import BasePolicy, PolicyScope

    @register_policy("my_policy")
    class MyPolicy(BasePolicy):
        @property
        def scope(self) -> PolicyScope:
            return PolicyScope.LOCAL_ONLY

        def choose_dc(self, dc_view, region_id, **kwargs):
            ...
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import inference_pb2
from common.network import SCHEDULER_GROUPS


class PolicyScope(Enum):
    LOCAL_ONLY = auto()
    GLOBAL = auto()
    GLOBAL_WITH_LOCAL = auto()


class BasePolicy(ABC):
    """Abstract base for all scheduler routing policies.

    Attributes:
        scheduler_group: Name of this scheduler's geographic group
            (e.g. "europe_north", "us_east").
        _local_regions: Set of datacenter IDs belonging to this group.
    """

    def __init__(self, scheduler_group: str = "europe_north", **kwargs):
        self.scheduler_group = scheduler_group
        self._local_regions = set(SCHEDULER_GROUPS.get(scheduler_group, []))

    @property
    @abstractmethod
    def scope(self) -> PolicyScope:
        """Declare the dispatch scope of this policy."""
        ...

    @abstractmethod
    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        """Pick a datacenter ID, or None if no valid candidate exists.

        Args:
            dc_view: Snapshot of all datacenter stats from the provisioner.
            region_id: This scheduler's group name.
            **kwargs: Extra data (e.g. rtt_matrix) — policies may ignore.
        """
        ...

    def choose_dc_extended(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[Tuple[str, bool]]:
        """Pick a datacenter, returning ``(dc_id, is_local)``.

        Must be overridden by GLOBAL and GLOBAL_WITH_LOCAL policies.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} (scope={self.scope.name}) "
            f"must implement choose_dc_extended()"
        )

    def choose_dc_local_only(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        """Pick from this group's datacentres only, for a forwarded request.

        The default restricts the view and then defers to choose_dc, so a policy
        gets local-only behaviour for free and only overrides this to score
        local candidates differently -- which is what carbon_aware does.

        It used to defer to choose_dc on the FULL view, which for a GLOBAL policy
        means choosing from every datacentre again. A request forwarded from
        europe_north to asia could then be dispatched to south_america, and the
        server marks the forwarded path local unconditionally, so the second hop
        cost nothing: 86% of forwarded requests in a smoke test landed outside
        both schedulers in their chain. Forwarding looked cheaper than it is, in
        the direction that encourages more of it.
        """
        local_view = {dc_id: stats for dc_id, stats in dc_view.items()
                      if dc_id in self._local_regions}
        if not local_view:
            return None
        return self.choose_dc(local_view, region_id, **kwargs)

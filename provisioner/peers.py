"""Who the fleet is, and how to reach them.

Discovery answers the first question and channel caching the second. Both are
here rather than inside the polling loop because they are the only part of the
provisioner that knows a peer is a network address at all: everything above
this file works in datacentre ids.

A discovery that fails is a warning, not an error. The previous map stays and
the loop keeps working against the fleet it last knew about, which is right for
a transient failure of the discovery backend and is why the loop above cannot
tell the difference.
"""

import logging
from typing import Dict

import grpc

import inference_pb2_grpc

from common.discovery import discover_datacenters, discover_schedulers

log = logging.getLogger("provisioner.peers")


class PeerRegistry:
    """The datacentres and schedulers this provisioner talks to."""

    def __init__(self) -> None:
        self.dc_endpoints: Dict[str, str] = {}
        self.sched_endpoints: Dict[str, str] = {}
        self._dc_channels: Dict[str, grpc.Channel] = {}
        self._sched_channels: Dict[str, grpc.Channel] = {}

    def refresh(self):
        try:
            self.dc_endpoints = discover_datacenters()
            log.info("DC discovery: %d datacenters", len(self.dc_endpoints))
        except Exception as e:
            log.warning("DC discovery failed: %s", e)

        try:
            self.sched_endpoints = discover_schedulers()
            log.info("Scheduler discovery: %d schedulers", len(self.sched_endpoints))
        except Exception as e:
            log.warning("Scheduler discovery failed: %s", e)

    def dc_stub(self, dc_id: str) -> inference_pb2_grpc.DatacenterServiceStub:
        addr = self.dc_endpoints[dc_id]
        if dc_id not in self._dc_channels:
            self._dc_channels[dc_id] = grpc.insecure_channel(addr)
        return inference_pb2_grpc.DatacenterServiceStub(self._dc_channels[dc_id])

    def sched_stub(self, sid: str) -> inference_pb2_grpc.SchedulerServiceStub:
        addr = self.sched_endpoints[sid]
        if sid not in self._sched_channels:
            self._sched_channels[sid] = grpc.insecure_channel(addr)
        return inference_pb2_grpc.SchedulerServiceStub(self._sched_channels[sid])

    def close(self) -> None:
        """Drop every channel. Only for shutdown; the loop reuses them."""
        for ch in list(self._dc_channels.values()) + list(self._sched_channels.values()):
            try:
                ch.close()
            except Exception:
                pass
        self._dc_channels.clear()
        self._sched_channels.clear()

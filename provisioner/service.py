"""The gRPC surface. A shim over the loop, and read-only.

Nothing in the repo calls either of these. They exist for a person looking at a
running fleet, which is what GetStatus is good for.

GetCarbonReport is a second, worse implementation of what
`scripts/sweep/collect_energy.py` does correctly: it polls sites one at a time rather
than fanning out, and it reads energy fields that stay zero until
`FinalizeEnergy` has run, which it never calls -- so it hands back zeros a
caller could believe. Use collect_energy.py. This is kept only because removing
an RPC from the proto is a decision about the interface, not a cleanup.
"""

import logging

import inference_pb2
import inference_pb2_grpc

log = logging.getLogger("provisioner.service")


class ProvisionerServicer(inference_pb2_grpc.ProvisionerServiceServicer):
    def __init__(self, prov):
        self.prov = prov

    def GetStatus(self, request, context):
        return inference_pb2.ProvisionerStatus(
            current_epoch=self.prov.epoch,
            num_datacenters=len(self.prov.dc_endpoints),
            num_schedulers=len(self.prov.sched_endpoints),
            poll_period_s=self.prov.poll_period_s,
            last_poll_duration=self.prov.last_poll_duration,
        )

    def GetCarbonReport(self, request, context):
        """Poll all DCs for current energy/carbon stats and return a report.

        Sequential, and zero before FinalizeEnergy. See the module docstring.
        """
        dc_stats_list = []
        total_energy = 0.0
        total_carbon = 0.0

        for dc_id in sorted(self.prov.dc_endpoints):
            try:
                resp = self.prov.poller.poll_one(dc_id, self.prov.epoch)
                dc_stats_list.append(inference_pb2.DcCarbonStats(
                    dc_id=resp.dc_id,
                    carbon_intensity=resp.carbon_intensity,
                    total_energy_j=resp.total_energy_j,
                    carbon_emissions_g=resp.carbon_emissions_g,
                ))
                total_energy += resp.total_energy_j
                total_carbon += resp.carbon_emissions_g
            except Exception as e:
                log.warning("Carbon poll %s failed: %s", dc_id, e)

        return inference_pb2.CarbonReport(
            dc_stats=dc_stats_list,
            total_energy_j=total_energy,
            total_carbon_emissions_g=total_carbon,
        )

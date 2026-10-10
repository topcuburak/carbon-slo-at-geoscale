"""Entry point: build the provisioner, start its loop, serve the gRPC surface.

    python -m provisioner.server              # driven by the environment
    provisioner/start.sh              # the same, with flags and checks

Everything substantive is elsewhere -- `peers`, `polling`, `congestion`,
`snapshot` and `loop`. This file only reads the environment and wires them up.
"""

import logging
import os

from common.threads import set_stack_size
from common.grpc_bind import bind_with_retry
import threading
from concurrent import futures

import grpc

import inference_pb2_grpc

from common.config import (
    PROVISIONER_GRPC_PORT,
    POLL_PERIOD_S,
    DISCOVERY_REFRESH_S,
    RTT_BETA,
)
from provisioner.loop import Provisioner
from provisioner.service import ProvisionerServicer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("provisioner")


def serve():
    _kb = set_stack_size()
    if _kb:
        log.info("thread stack %d KB (SIM_THREAD_STACK_KB)", _kb)
    poll_s = float(os.environ.get("POLL_PERIOD_S", str(POLL_PERIOD_S)))
    disc_s = float(os.environ.get("DISCOVERY_REFRESH_S", str(DISCOVERY_REFRESH_S)))
    rtt_beta = float(os.environ.get("RTT_BETA", str(RTT_BETA)))
    rtt_norm = os.environ.get("RTT_NORM", "cap").strip().lower()
    rtt_heat_cap = float(os.environ.get("RTT_HEAT_CAP", "32"))
    port = int(os.environ.get("GRPC_PORT", str(PROVISIONER_GRPC_PORT)))

    poll_mode = os.environ.get("POLL_MODE", "sync").strip().lower()
    wave_count = int(os.environ.get("POLL_WAVE_COUNT", "4"))
    wave_interval_s = float(os.environ.get("POLL_WAVE_INTERVAL_S", "0.5"))

    prov = Provisioner(
        poll_s, disc_s, rtt_beta=rtt_beta,
        rtt_norm=rtt_norm, rtt_heat_cap=rtt_heat_cap,
        poll_mode=poll_mode,
        wave_count=wave_count,
        wave_interval_s=wave_interval_s,
    )

    poll_thread = threading.Thread(target=prov.run, daemon=True)
    poll_thread.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    inference_pb2_grpc.add_ProvisionerServiceServicer_to_server(
        ProvisionerServicer(prov), server,
    )
    bind_with_retry(server, port)
    server.start()
    if poll_mode == "async":
        log.info(
            "Provisioner listening on port %d  mode=async  waves=%d  "
            "wave_interval=%.2fs  (cycle=%.1fs)  discovery_refresh=%.1fs  "
            "rtt_beta=%.2f  rtt_norm=%s(cap=%.0f)",
            port, wave_count, wave_interval_s,
            wave_count * wave_interval_s, disc_s, rtt_beta,
            rtt_norm, rtt_heat_cap,
        )
    else:
        log.info(
            "Provisioner listening on port %d  mode=sync  poll=%.1fs  "
            "discovery_refresh=%.1fs  rtt_beta=%.2f  rtt_norm=%s(cap=%.0f)",
            port, poll_s, disc_s, rtt_beta, rtt_norm, rtt_heat_cap,
        )

    try:
        server.wait_for_termination()
    finally:
        prov.stop()
        poll_thread.join(timeout=2.0)
        prov.peers.close()


if __name__ == "__main__":
    serve()

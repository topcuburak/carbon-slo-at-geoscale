"""Shared configuration for all microservices.

Every value can be overridden via the environment variable shown in the
``os.environ.get()`` call.  Defaults match the original Python simulation.
"""

import os
import json

# ---------- simulation model ----------
SERVICE_MS_PER_TOK: float = float(os.environ.get("SERVICE_MS_PER_TOK", "0.08"))

# ---------- provisioner ----------
POLL_PERIOD_S: float = float(os.environ.get("POLL_PERIOD_S", "2"))
DISCOVERY_REFRESH_S: float = float(os.environ.get("DISCOVERY_REFRESH_S", "10.0"))

# ---------- datacenter ----------
NUM_NODES_PER_DC: int = int(os.environ.get("NUM_NODES_PER_DC", "2"))
NUM_GPUS_PER_NODE: int = int(os.environ.get("NUM_GPUS_PER_NODE", "8"))
TP_SIZE: int = int(os.environ.get("TP_SIZE", "2"))

# ---------- carbon-aware scheduling ----------
CARBON_INTENSITY: float = float(os.environ.get("CARBON_INTENSITY", "0.0"))
SCHEDULER_GROUP: str = os.environ.get("SCHEDULER_GROUP", "europe")
ROUTING_ALPHA: float = float(os.environ.get("ROUTING_ALPHA", "0.5"))
SIMULATE_RTT: bool = os.environ.get("SIMULATE_RTT", "true").lower() in ("true", "1", "yes")
RTT_BETA: float = float(os.environ.get("RTT_BETA", "0.5"))
MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "0"))  # 0 = unlimited
PUE: float = float(os.environ.get("PUE", "1.1"))  # Power Usage Effectiveness
REF_ISL: int = int(os.environ.get("REF_ISL", "7500"))
REF_OSL: int = int(os.environ.get("REF_OSL", "500"))
# Where the scheduler gets a request's output length when it prices a
# placement: "oracle" (the trace's true out_tok, which makes it clairvoyant),
# "predicted" (a length predictor's output carried on the request), or "ref"
# (the flat REF_OSL above). See scheduler/policies/latency_estimates.py.
OSL_SOURCE: str = os.environ.get("OSL_SOURCE", "oracle")

# ---------- networking ----------
DATACENTER_GRPC_PORT: int = int(os.environ.get("DATACENTER_GRPC_PORT", "50051"))
SCHEDULER_GRPC_PORT: int = int(os.environ.get("SCHEDULER_GRPC_PORT", "50052"))
PROVISIONER_GRPC_PORT: int = int(os.environ.get("PROVISIONER_GRPC_PORT", "50053"))

# ---------- K8s labels used for service discovery ----------
LABEL_COMPONENT_KEY = "app.kubernetes.io/component"
LABEL_DC_COMPONENT = "datacenter"
LABEL_SCHED_COMPONENT = "scheduler"
LABEL_DC_ID_KEY = "dc-id"
LABEL_SCHED_ID_KEY = "sched-id"

# ---------- topology ----------
# Region → list of datacenter IDs.  Loaded from REGION_MAP_JSON env var
# or a ConfigMap-mounted file.  Fall back to a simple default.
_DEFAULT_REGION_MAP = {
    f"region{i}": [f"dc-{i * 3 + j}" for j in range(3)]
    for i in range(5)
}

def load_region_map() -> dict:
    raw = os.environ.get("REGION_MAP_JSON")
    if raw:
        return json.loads(raw)
    path = os.environ.get("REGION_MAP_FILE")
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return _DEFAULT_REGION_MAP


# ---------- static endpoint overrides (for non-K8s testing) ----------
def load_static_endpoints() -> dict | None:
    """Return {dc_id: 'host:port', ...} or None to use K8s discovery."""
    raw = os.environ.get("STATIC_ENDPOINTS_JSON")
    if raw:
        return json.loads(raw)
    path = os.environ.get("STATIC_ENDPOINTS_FILE")
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None

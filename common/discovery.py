"""Cross-namespace Kubernetes service discovery.

Uses the ``kubernetes`` Python client to list Services by label selector,
producing a ``{component_id: "host:port"}`` endpoint map.

Falls back to static endpoint maps when running outside a cluster
(controlled via env-var, see config.py).
"""

import logging
import os
from typing import Dict, Optional

from common.config import (
    LABEL_COMPONENT_KEY,
    LABEL_DC_COMPONENT,
    LABEL_DC_ID_KEY,
    LABEL_SCHED_COMPONENT,
    LABEL_SCHED_ID_KEY,
    DATACENTER_GRPC_PORT,
    SCHEDULER_GRPC_PORT,
    PROVISIONER_GRPC_PORT,
    load_static_endpoints,
)

log = logging.getLogger(__name__)

def _k8s_client():
    """Return a CoreV1Api, loading in-cluster or kubeconfig as appropriate."""
    from kubernetes import client, config as kube_config

    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        kube_config.load_incluster_config()
    else:
        kube_config.load_kube_config()
    return client.CoreV1Api()


def discover_datacenters(
    static_override: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return ``{dc_id: 'host:port'}`` for every datacenter service.

    If *static_override* is provided (or the env-var fallback is set),
    skip K8s API calls entirely.
    """
    endpoints = static_override or load_static_endpoints()
    if endpoints is not None:
        # Filter non-datacentre entries out of mixed static endpoint files.
        # "provisioner" used to survive this and be polled as if it were a
        # site, so every provisioner.log carried one "Poll provisioner failed"
        # per cycle forever -- 38 of them in a 15 s run. Harmless to placement
        # (a failed poll never reaches a snapshot) but it buries the warning
        # that means something: a real datacentre that did not come up.
        dc_eps = {k: v for k, v in endpoints.items()
                  if not k.startswith("sched") and k != "provisioner"}
        log.info("Using static datacenter endpoints (%d entries)", len(dc_eps))
        return dc_eps

    v1 = _k8s_client()
    label = f"{LABEL_COMPONENT_KEY}={LABEL_DC_COMPONENT}"
    services = v1.list_service_for_all_namespaces(label_selector=label)

    result: Dict[str, str] = {}
    for svc in services.items:
        dc_id = svc.metadata.labels.get(LABEL_DC_ID_KEY)
        if not dc_id:
            continue
        ns = svc.metadata.namespace
        name = svc.metadata.name
        port = DATACENTER_GRPC_PORT
        for p in svc.spec.ports:
            if p.name == "grpc":
                port = p.port
                break
        result[dc_id] = f"{name}.{ns}.svc.cluster.local:{port}"

    log.info("Discovered %d datacenter(s) via K8s API", len(result))
    return result

def discover_schedulers(
    static_override: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return ``{sched_id: 'host:port'}`` for every scheduler service."""
    endpoints = static_override or load_static_endpoints()
    if endpoints is not None:
        # Static map may contain mixed entries; filter by prefix convention
        return {k: v for k, v in endpoints.items() if k.startswith("sched")}

    v1 = _k8s_client()
    label = f"{LABEL_COMPONENT_KEY}={LABEL_SCHED_COMPONENT}"
    services = v1.list_service_for_all_namespaces(label_selector=label)

    result: Dict[str, str] = {}
    for svc in services.items:
        sid = svc.metadata.labels.get(LABEL_SCHED_ID_KEY)
        if not sid:
            continue
        ns = svc.metadata.namespace
        name = svc.metadata.name
        port = SCHEDULER_GRPC_PORT
        for p in svc.spec.ports:
            if p.name == "grpc":
                port = p.port
                break
        result[sid] = f"{name}.{ns}.svc.cluster.local:{port}"

    log.info("Discovered %d scheduler(s) via K8s API", len(result))
    return result


def discover_provisioner(
    static_override: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return ``'host:port'`` for the provisioner, or None if not found."""
    endpoints = static_override or load_static_endpoints()
    if endpoints is not None:
        addr = endpoints.get("provisioner")
        if addr:
            log.info("Using static provisioner endpoint: %s", addr)
            return addr
        return None

    # K8s discovery fallback
    v1 = _k8s_client()
    label = f"{LABEL_COMPONENT_KEY}=provisioner"
    services = v1.list_service_for_all_namespaces(label_selector=label)
    for svc in services.items:
        ns = svc.metadata.namespace
        name = svc.metadata.name
        port = PROVISIONER_GRPC_PORT
        for p in svc.spec.ports:
            if p.name == "grpc":
                port = p.port
                break
        addr = f"{name}.{ns}.svc.cluster.local:{port}"
        log.info("Discovered provisioner via K8s API: %s", addr)
        return addr

    return None

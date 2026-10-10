"""Load scheduler configuration from YAML with environment variable overrides.

Priority: Environment Variable > YAML file > Hardcoded defaults.

Usage::

    from scheduler.config_loader import load_scheduler_config

    cfg = load_scheduler_config()           # auto-discovers config.yaml
    cfg = load_scheduler_config("my.yaml")  # explicit path
"""

import os
from pathlib import Path

import yaml


# Hardcoded defaults (fallback when neither env var nor YAML provides a value)
_DEFAULTS = {
    "sched_id": "sched-europe",
    "scheduler_group": "europe",
    "routing_policy": "carbon_aware",
    "routing_alpha": 0.5,
    "simulate_rtt": True,
    "grpc_port": 50052,
    "grpc_max_workers": 5000,
    "verbosity": 1,
    "log_tag": "",
    "static_endpoints_file": None,
    # Batch ILP settings
    # MAX wait before an ILP batch is solved (ms). 100, not 250: the batch
    # loop runs on an absolute cadence, so the cycle is max(window, solve),
    # and the solve is ~118 ms at 600 req/s per scheduler. Below 100 the
    # window stops being the binding term and buys nothing. Measured
    # 2026-09-08: at 250 the ILP carried scheduling_ms p50 243 ms against a
    # 3000 ms budget, which was its whole SLO deficit against the greedy arm.
    "batch_window_ms": 100,
    # Requests that wake the batch early. A TRIGGER, not a drain cap: the loop
    # still takes everything buffered, because capping the drain would turn a
    # throughput deficit into an unbounded backlog rather than fixing it.
    # 300 is derived, not picked -- at the measured 27 us per (request, DC)
    # cell and 16 sites, 300 requests is 130 ms of build, just over half the
    # 250 ms window. 500 would be 216 ms, i.e. 86% of it, which is the
    # utilisation where jitter starts queueing and is the regime that produced
    # the original collapse.
    "batch_window_size": 300,
    "ttft_budget_ms": 0,          # SLA: additive overhead slack in ms
                                   #      (full budget = slack + 1.5·prefill_ms(ISL))
    "tpot_budget_ms": 0,          # SLA: flat TPOT budget in ms
    "bs_cap": 0,                  # Hard per-DC load cap (active+pending+new)
                                   #      0 disables; paper default 64.
    "ilp_max_iterations": 3,      # Max rounds for iterative ILP
    "model_name": "",             # Model name for analytical latency model
    "tp_size": 1,                 # Tensor parallel size
    # The engines' max_num_seqs, as the datacentres are configured. Gates the
    # admission-wait term in the TTFT estimate: below the cap a request is
    # admitted at once, at the cap it waits for a sequence slot. 0 = unknown,
    # and the term then applies unconditionally.
    "max_num_seqs": 0,
    # Where a placement decision gets the request's output length:
    # "oracle" | "predicted" | "ref". See latency_estimates.OSL_SOURCES.
    "osl_source": "oracle",
    "ref_osl": 500,
    # Schedulers dispatching into the same fleet. The view projection scales
    # this scheduler's own in-flight work by it, on the symmetric assumption
    # that the others are doing the same. 1 = project only my own.
    "peer_scale": 1,
}

# Mapping: flat key -> (env var name, YAML path as tuple of keys)
_PARAM_MAP = {
    "sched_id":              ("SCHED_ID",              ("scheduler", "sched_id")),
    "scheduler_group":       ("SCHEDULER_GROUP",       ("scheduler", "scheduler_group")),
    "routing_policy":        ("ROUTING_POLICY",        ("scheduler", "routing", "policy")),
    "routing_alpha":         ("ROUTING_ALPHA",          ("scheduler", "routing", "alpha")),
    "simulate_rtt":          ("SIMULATE_RTT",           ("scheduler", "routing", "simulate_rtt")),
    "grpc_port":             ("GRPC_PORT",              ("scheduler", "server", "grpc_port")),
    "grpc_max_workers":      ("GRPC_MAX_WORKERS",       ("scheduler", "server", "grpc_max_workers")),
    "verbosity":             ("VERBOSITY",              ("scheduler", "logging", "verbosity")),
    "log_tag":               ("LOG_TAG",                ("scheduler", "logging", "log_tag")),
    "static_endpoints_file": ("STATIC_ENDPOINTS_FILE",  ("scheduler", "discovery", "static_endpoints_file")),
    "batch_window_ms":       ("BATCH_WINDOW_MS",        ("scheduler", "batch", "window_ms")),
    "batch_window_size":     ("BATCH_WINDOW_SIZE",      ("scheduler", "batch", "window_size")),
    "ttft_budget_ms":        ("TTFT_BUDGET_MS",          ("scheduler", "sla", "ttft_budget_ms")),
    "tpot_budget_ms":        ("TPOT_BUDGET_MS",          ("scheduler", "sla", "tpot_budget_ms")),
    "bs_cap":                ("BS_CAP",                   ("scheduler", "sla", "bs_cap")),
    "ilp_max_iterations":    ("ILP_MAX_ITERATIONS",       ("scheduler", "ilp", "max_iterations")),
    "model_name":            ("MODEL_NAME",               ("scheduler", "model_name")),
    "tp_size":               ("TP_SIZE",                   ("scheduler", "tp_size")),
    "max_num_seqs":          ("MAX_BATCH_SIZE",            ("scheduler", "max_num_seqs")),
    "osl_source":            ("OSL_SOURCE",                ("scheduler", "osl_source")),
    "ref_osl":               ("REF_OSL",                   ("scheduler", "ref_osl")),
    "peer_scale":            ("SCHED_PEER_SCALE",          ("scheduler", "peer_scale")),
}


def _dig(data: dict, keys: tuple):
    """Traverse nested dict by key path. Returns None if any key is missing."""
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data


def _cast(key: str, raw: str):
    """Cast a string env-var value to the expected Python type."""
    default = _DEFAULTS[key]
    if isinstance(default, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _find_config_file(config_path=None) -> Path | None:
    """Locate the YAML config file.

    Search order:
      1. Explicit ``config_path`` argument
      2. ``SCHEDULER_CONFIG`` environment variable
      3. ``scheduler/config.yaml`` relative to project root
    """
    if config_path:
        p = Path(config_path)
        return p if p.is_file() else None

    env_path = os.environ.get("SCHEDULER_CONFIG")
    if env_path:
        p = Path(env_path)
        return p if p.is_file() else None

    # Default: scheduler/config.yaml next to this file
    default = Path(__file__).parent / "config.yaml"
    return default if default.is_file() else None


def load_scheduler_config(config_path=None) -> dict:
    """Load scheduler configuration.

    Returns a flat dict with these keys::

        sched_id, scheduler_group, routing_policy, routing_alpha,
        simulate_rtt, grpc_port, grpc_max_workers, verbosity,
        log_tag, static_endpoints_file
    """
    # 1. Start with hardcoded defaults
    cfg = dict(_DEFAULTS)

    # 2. Override with YAML values (if config file exists)
    yaml_path = _find_config_file(config_path)
    yaml_data = {}
    if yaml_path:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    for key, (_, yaml_keys) in _PARAM_MAP.items():
        val = _dig(yaml_data, yaml_keys)
        if val is not None:
            cfg[key] = val

    # 3. Override with environment variables (highest priority)
    for key, (env_var, _) in _PARAM_MAP.items():
        env_val = os.environ.get(env_var)
        if env_val is not None:
            cfg[key] = _cast(key, env_val)

    return cfg

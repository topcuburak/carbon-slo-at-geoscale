#!/usr/bin/env python3
"""Run one experiment family from configs/*.json.

Each JSON in configs/ is the configuration a figure in the paper was produced
with. This script maps its fields onto the environment variables read by
scripts/sweep/sweep_all_policies.sh and launches the sweep.

    python scripts/reproduce/run_config.py configs/fig4_llama_ilp_rr.json \
        --results results/fig4_llama [--policies "round_robin facility_ilp_carbon_sla"] \
        [--rates "1200 2400 4000"] [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys

# config field -> sweep environment variable
FIELDS = {
    "seed": "SEED", "sim_datetime": "SIM_DATETIME", "duration": "DURATION",
    "ttft_budget_ms": "TTFT_BUDGET", "tpot_budget_ms": "TPOT_BUDGET",
    "tpot_filter": "TPOT_FILTER", "peer_scale": "SCHED_PEER_SCALE",
    "bs_cap": "BS_CAP", "poll_mode": "POLL_MODE", "poll_period_s": "POLL_PERIOD_S",
    "poll_wave_count": "POLL_WAVE_COUNT", "poll_wave_interval_s": "POLL_WAVE_INTERVAL_S",
    "model_name": "MODEL_NAME", "tp_size": "TP_SIZE", "num_nodes": "NUM_NODES",
    "num_gpus": "NUM_GPUS", "max_batch_size": "MAX_BATCH_SIZE",
    "max_model_len": "MAX_MODEL_LEN", "osl_source": "OSL_SOURCE", "ref_osl": "REF_OSL",
    "batch_window_ms": "BATCH_WINDOW_MS", "batch_window_size": "BATCH_WINDOW_SIZE",
    "loadgen_stream": "LOADGEN_STREAM", "loadgen_shards": "LOADGEN_SHARDS",
    "ilp_solve_process": "ILP_SOLVE_PROCESS", "pue_table": "PUE_TABLE",
    "kv_capacity_source": "KV_CAPACITY_SOURCE", "kv_queue_agg": "KV_QUEUE_AGG",
    "topology_json": "TOPOLOGY_JSON", "trace_csv": "TRACE_CSV",
}
# Settings used for every run in the paper that are not recorded in config.json.
FIXED = {
    "SCHED_PEER_STREAM": "1", "SCHED_PEER_PROJECT": "1", "FWD_ARRIVAL_RECHECK": "1",
    "DC_LATENCY_MODE": "analytical", "ERR_ABORT_THRESHOLD": "101",
    "SIM_TASKS_MAX": "250000", "GRPC_MAX_WORKERS": "50000",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--results", required=True, help="output directory (RESULTS_ROOT)")
    ap.add_argument("--policies", help="space-separated subset of the config's policies")
    ap.add_argument("--rates", help="space-separated subset of the config's arrival rates")
    ap.add_argument("--dry-run", action="store_true", help="print the environment only")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    env = dict(os.environ, **FIXED)
    for key, var in FIELDS.items():
        if key in cfg:
            env[var] = str(cfg[key])
    env["POLICIES"] = a.policies or " ".join(cfg["policies"])
    env["ARRIVAL_RATES"] = a.rates or " ".join(str(r) for r in cfg["arrival_rates"])
    env["RESULTS_ROOT"] = a.results

    shown = sorted(set(FIELDS.values()) | set(FIXED) | {"POLICIES", "ARRIVAL_RATES", "RESULTS_ROOT"})
    for var in shown:
        if var in env:
            print(f"{var}={env[var]}")
    if a.dry_run:
        return 0
    return subprocess.call(["bash", "scripts/sweep/sweep_all_policies.sh"], env=env)


if __name__ == "__main__":
    sys.exit(main())

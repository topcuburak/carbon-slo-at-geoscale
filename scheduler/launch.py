"""Start one scheduler per group and wait until the fleet is reachable.

How many schedulers there are is NOT a free parameter. Forwarding resolves the
owner of a datacentre through `common.network.REGION_TO_GROUP` and then builds
the peer's identity as `sched-<group>`, so a group without a process, or a
process under a different id, is not a smaller fleet -- it is a fleet where
every request for that group's datacentres dies with "no scheduler for DC".
With 104 datacentres across 8 groups, missing one silently kills up to 26
regions' traffic. So the group list comes from the topology, the ids follow the
convention, and this script refuses to hand back a fleet with a hole in it.

    python -m scheduler.launch --policy carbon_aware --alpha 0.5 \
        --model-name llama3_8B --tp-size 2
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from common.network import SCHEDULER_GROUPS


def port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="all",
                    help="'all' (the topology's own list) or a comma-separated "
                         "subset. A subset is a deliberately partial fleet: "
                         "requests for the groups left out will fail.")
    ap.add_argument("--base-port", type=int, default=50200)
    ap.add_argument("--policy", default="round_robin")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--ilp-max-iterations", type=int, default=3,
                    help="rounds for iterative_ilp_carbon_sla; "
                         "ignored by the other policies")
    ap.add_argument("--model-name", default="",
                    help="required by every policy that estimates latency; "
                         "without it those refuse to start")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--max-num-seqs", type=int, default=32,
                    help="the engines' PER-ENGINE max_num_seqs, as the "
                         "datacentres are configured. A site's capacity is "
                         "this times its engine count. The ILP splits its "
                         "projected placements into admitted and queued "
                         "against it; left at 0 every placement projects as "
                         "admitted, which is not what the sweep does.")
    ap.add_argument("--simulate-rtt", default="true")
    ap.add_argument("--verbosity", type=int, default=1)
    ap.add_argument("--batch-window-ms", type=int, default=250)
    ap.add_argument("--ttft-budget-ms", type=int, default=0)
    ap.add_argument("--tpot-budget-ms", type=int, default=0)
    ap.add_argument("--osl-source", default="oracle")
    ap.add_argument("--ref-osl", type=int, default=500)
    ap.add_argument("--endpoints-file", default="results/logs/endpoints.json")
    ap.add_argument("--log-dir", default="results/logs/schedulers")
    ap.add_argument("--log-tag", default="")
    ap.add_argument("--wait-s", type=float, default=30.0)
    a = ap.parse_args()

    groups = (list(SCHEDULER_GROUPS) if a.groups == "all"
              else [g.strip() for g in a.groups.split(",") if g.strip()])
    unknown = [g for g in groups if g not in SCHEDULER_GROUPS]
    if unknown:
        sys.exit(f"unknown group(s): {', '.join(unknown)}. "
                 f"Known: {', '.join(SCHEDULER_GROUPS)}")
    if a.groups != "all":
        missing = [g for g in SCHEDULER_GROUPS if g not in groups]
        print(f"  PARTIAL FLEET: {len(missing)} group(s) not started "
              f"({', '.join(missing)}). Requests for their datacentres will "
              f"fail; this is only correct if the run also restricts the DCs.",
              file=sys.stderr)

    ports = {g: a.base_port + i for i, g in enumerate(SCHEDULER_GROUPS)
             if g in groups}

    # The endpoint map has to exist before the processes do: a scheduler reads
    # it at startup to learn its peers, and a peer that is not in the map cannot
    # be forwarded to even once it is running.
    endpoints = {f"sched-{g}": f"localhost:{p}" for g, p in ports.items()}
    ep_path = Path(a.endpoints_file)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if ep_path.is_file():
        try:
            existing = json.loads(ep_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    # Keep whatever else is in there (the datacentres, the provisioner) and
    # replace only the scheduler entries.
    merged = {k: v for k, v in existing.items() if not k.startswith("sched-")}
    merged.update(endpoints)
    ep_path.write_text(json.dumps(merged, indent=2) + "\n")

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.log_tag}_" if a.log_tag else ""

    procs = {}
    for g, port in ports.items():
        env = dict(os.environ)
        env.update({
            # sched-<group> is not a naming preference: _find_sched_for_dc
            # builds exactly this string to address a peer.
            "SCHED_ID": f"sched-{g}",
            "SCHEDULER_GROUP": g,
            "ROUTING_POLICY": a.policy,
            "ROUTING_ALPHA": str(a.alpha),
            "ILP_MAX_ITERATIONS": str(a.ilp_max_iterations),
            "SIMULATE_RTT": a.simulate_rtt,
            "GRPC_PORT": str(port),
            "STATIC_ENDPOINTS_FILE": str(ep_path),
            "TP_SIZE": str(a.tp_size),
            # config_loader reads it under this name, the same one the
            # datacentres take.
            "MAX_BATCH_SIZE": str(a.max_num_seqs),
            "VERBOSITY": str(a.verbosity),
            "BATCH_WINDOW_MS": str(a.batch_window_ms),
            "OSL_SOURCE": a.osl_source,
            "REF_OSL": str(a.ref_osl),
        })
        if a.model_name:
            env["MODEL_NAME"] = a.model_name
        if a.ttft_budget_ms:
            env["TTFT_BUDGET_MS"] = str(a.ttft_budget_ms)
        if a.tpot_budget_ms:
            env["TPOT_BUDGET_MS"] = str(a.tpot_budget_ms)

        logf = open(log_dir / f"{tag}sched_{g}.log", "w")
        procs[g] = subprocess.Popen(
            [sys.executable, "-m", "scheduler.server"],
            env=env, stdout=logf, stderr=subprocess.STDOUT)

    # Wait for every one, and say which failed rather than leaving a hole to be
    # discovered later as per-request errors.
    deadline = time.time() + a.wait_s
    pending = dict(ports)
    while pending and time.time() < deadline:
        for g, port in list(pending.items()):
            if port_open(port):
                del pending[g]
        if pending:
            time.sleep(0.25)

    for g in ports:
        rc = procs[g].poll()
        if rc is not None:
            tail = (log_dir / f"{tag}sched_{g}.log").read_text().strip().splitlines()
            print(f"  sched-{g} EXITED rc={rc}: {tail[-1] if tail else ''}",
                  file=sys.stderr)

    if pending:
        for p in procs.values():
            p.terminate()
        sys.exit(f"{len(pending)} scheduler(s) never came up: "
                 f"{', '.join(sorted(pending))}. See {log_dir}/. "
                 f"Started none of the fleet rather than a partial one.")

    print(f"  {len(ports)} schedulers up on ports "
          f"{min(ports.values())}-{max(ports.values())}  "
          f"(policy={a.policy} alpha={a.alpha} model={a.model_name or 'NONE'})")
    print(f"  endpoints: {ep_path}")
    for g, port in ports.items():
        print(f"      sched-{g:<14} localhost:{port}  "
              f"{len(SCHEDULER_GROUPS[g])} DCs")


if __name__ == "__main__":
    main()

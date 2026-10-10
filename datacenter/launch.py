"""Start one datacentre process per site and wait until all of them answer.

Which sites exist is not a free parameter. `common.network.SCHEDULER_GROUPS` is
the topology -- 104 datacentres across 8 groups -- and a scheduler resolves a
site's owner through it, so a site that is in the topology but has no process is
not a smaller fleet: every request routed to it dies. This starts the whole list
by default and says loudly when it is asked for a subset.

Each site gets three things that are properly per-site and used to be global or
hand-passed:

  carbon intensity  from `common.carbon.CarbonLoader` at `--sim-datetime`, the
                    same source and timestamp the sweep uses. A site with no
                    data in the archive is reported, not silently defaulted.
  PUE               from `common/pue.py`, which reads common/pue_by_dc.json.
  port              assigned in topology order from `--base-port`.

    python -m datacenter.launch
    python -m datacenter.launch --model-name qwen2.5_7B --tp-size 2
    python -m datacenter.launch --groups europe_north --verbosity 2

Ctrl-C, SIGTERM, or `--stop` shuts the fleet down. Nothing is left behind: a
site that fails to bind its port takes the whole fleet down with it rather than
leaving a hole to be discovered later as per-request errors.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common.network import SCHEDULER_GROUPS
from common.pue import get_pue


def port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def resolve_dcs(groups_arg: str):
    """The datacentre list, in topology order, with the group each belongs to."""
    if groups_arg == "all":
        groups = list(SCHEDULER_GROUPS)
    else:
        groups = [g.strip() for g in groups_arg.split(",") if g.strip()]
        unknown = [g for g in groups if g not in SCHEDULER_GROUPS]
        if unknown:
            sys.exit(f"unknown group(s): {', '.join(unknown)}. "
                     f"Known: {', '.join(SCHEDULER_GROUPS)}")
    return [(dc, g) for g in SCHEDULER_GROUPS if g in groups
            for dc in SCHEDULER_GROUPS[g]]


def load_ci(dc_ids, when: str, fallback: float):
    """Carbon intensity per site, from the archive, at one timestamp.

    Constant for the length of a run on purpose: a run is around two minutes of
    simulated time and grid intensity does not move meaningfully inside that.
    Temporal variation is studied by running the same experiment at a different
    `--sim-datetime`, not by varying it mid-run.
    """
    try:
        from common.carbon import CarbonLoader
    except ImportError as e:
        print(f"  carbon archive unavailable ({e}); every site takes "
              f"--default-ci {fallback}", file=sys.stderr)
        return {d: fallback for d in dc_ids}, []
    try:
        loader = CarbonLoader("carbon_intensity", year=2024, regions=list(dc_ids))
        dt = datetime.fromisoformat(when)
    except Exception as e:
        print(f"  carbon archive unreadable ({e}); every site takes "
              f"--default-ci {fallback}", file=sys.stderr)
        return {d: fallback for d in dc_ids}, []
    ci, missing = {}, []
    for d in dc_ids:
        try:
            v = float(loader.get_ci(d, dt))
        except Exception:
            v, _ = fallback, missing.append(d)
        ci[d] = v
    return ci, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="all",
                    help="'all' (the topology's own list) or a comma-separated "
                         "subset. A subset is a deliberately partial fleet.")
    ap.add_argument("--base-port", type=int, default=51000)
    ap.add_argument("--num-nodes", type=int, default=2)
    ap.add_argument("--num-gpus-per-node", type=int, default=8,
                    help="physical cards per node; engines = this // tp-size")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--max-batch-size", type=int, default=32,
                    help="max_num_seqs per engine; 0 for unlimited")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--model-name", default="llama3_8B")
    ap.add_argument("--latency-mode", default="analytical",
                    choices=("analytical", "lookup"))
    ap.add_argument("--latency-csv", default="")
    ap.add_argument("--sim-datetime", default="2024-07-15T12:00:00",
                    help="when to sample carbon intensity from the archive")
    ap.add_argument("--default-ci", type=float, default=400.0,
                    help="used only for sites the archive has no data for")
    ap.add_argument("--endpoints-file", default="results/logs/endpoints.json")
    ap.add_argument("--log-dir", default="results/logs/datacenters")
    ap.add_argument("--log-tag", default="")
    ap.add_argument("--bs-profile-dir", default="")
    ap.add_argument("--verbosity", type=int, default=0)
    ap.add_argument("--wait-s", type=float, default=60.0)
    ap.add_argument("--foreground", action="store_true",
                    help="stay attached and shut the fleet down on Ctrl-C")
    a = ap.parse_args()

    pairs = resolve_dcs(a.groups)
    if not pairs:
        sys.exit("no datacentres selected")
    if a.groups != "all":
        left = sum(len(v) for g, v in SCHEDULER_GROUPS.items()
                   if g not in {g for _, g in pairs})
        print(f"  PARTIAL FLEET: {left} datacentre(s) not started. Requests "
              f"routed to them will fail; only correct if the run restricts "
              f"the schedulers too.", file=sys.stderr)

    dc_ids = [d for d, _ in pairs]
    ports = {d: a.base_port + i for i, d in enumerate(dc_ids)}
    ci, missing_ci = load_ci(dc_ids, a.sim_datetime, a.default_ci)
    if missing_ci:
        print(f"  {len(missing_ci)} site(s) have no carbon data at "
              f"{a.sim_datetime} and take --default-ci {a.default_ci}: "
              f"{', '.join(missing_ci[:8])}"
              f"{' ...' if len(missing_ci) > 8 else ''}", file=sys.stderr)

    # The endpoint map must exist before the processes do; a scheduler reads it
    # at startup to learn which sites it owns. Keep whatever else is in there
    # (the schedulers, the provisioner) and replace only the datacentre rows.
    endpoints = {d: f"localhost:{p}" for d, p in ports.items()}
    ep_path = Path(a.endpoints_file)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if ep_path.is_file():
        try:
            existing = json.loads(ep_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    merged = {k: v for k, v in existing.items()
              if k.startswith("sched-") or k == "provisioner"}
    merged.update(endpoints)
    ep_path.write_text(json.dumps(merged, indent=2) + "\n")

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.log_tag}_" if a.log_tag else ""

    procs, logs = {}, {}
    for dc_id in dc_ids:
        env = dict(os.environ)
        env.update({
            "DC_ID": dc_id,
            "GRPC_PORT": str(ports[dc_id]),
            "NUM_NODES": str(a.num_nodes),
            "NUM_GPUS_PER_NODE": str(a.num_gpus_per_node),
            "TP_SIZE": str(a.tp_size),
            "MAX_BATCH_SIZE": str(a.max_batch_size),
            "MAX_MODEL_LEN": str(a.max_model_len),
            "MODEL_NAME": a.model_name,
            "LATENCY_MODE": a.latency_mode,
            "CARBON_INTENSITY": f"{ci[dc_id]:.6f}",
            "VERBOSITY": str(a.verbosity),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(Path.cwd()), str(Path.cwd() / "proto"),
                              os.environ.get("PYTHONPATH", "")])),
        })
        # PUE is NOT exported. The site resolves its own from common/pue_by_dc.json
        # via common/pue.py, so the datacentre and scripts/sweep/collect_energy.py read
        # one table instead of two. Setting it here would override every site at
        # once, which is the opposite of per-site.
        env.pop("PUE", None)
        if a.latency_csv:
            env["LATENCY_CSV"] = a.latency_csv
        if a.bs_profile_dir:
            env["BS_PROFILE_DIR"] = a.bs_profile_dir

        logs[dc_id] = log_dir / f"{tag}dc_{dc_id}.log"
        procs[dc_id] = subprocess.Popen(
            [sys.executable, "-m", "datacenter.server"],
            env=env, stdout=open(logs[dc_id], "w"), stderr=subprocess.STDOUT)

    def shutdown(*_):
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    deadline = time.time() + a.wait_s
    pending = dict(ports)
    while pending and time.time() < deadline:
        for d, port in list(pending.items()):
            if port_open(port):
                del pending[d]
        if pending:
            time.sleep(0.25)

    # Say which died and why, rather than leaving a hole for the run to find.
    for d in dc_ids:
        rc = procs[d].poll()
        if rc is not None:
            tail = logs[d].read_text().strip().splitlines()
            print(f"  {d} EXITED rc={rc}: {tail[-1] if tail else ''}",
                  file=sys.stderr)
    if pending:
        shutdown()
        sys.exit(f"{len(pending)} datacentre(s) never came up: "
                 f"{', '.join(sorted(pending)[:8])}"
                 f"{' ...' if len(pending) > 8 else ''}. See {log_dir}/. "
                 f"Started none of the fleet rather than a partial one.")

    pues = sorted({round(get_pue(d), 6) for d in dc_ids})
    pue_str = (f"{pues[0]:.3f}" if len(pues) == 1
               else f"{pues[0]:.3f}-{pues[-1]:.3f} over {len(pues)} values")
    engines = a.num_nodes * (a.num_gpus_per_node // a.tp_size)
    print(f"  {len(dc_ids)} datacentres up on ports "
          f"{min(ports.values())}-{max(ports.values())}")
    print(f"      model={a.model_name} tp={a.tp_size} "
          f"nodes={a.num_nodes} cards/node={a.num_gpus_per_node} "
          f"-> {engines} engines/site, max_num_seqs={a.max_batch_size}")
    print(f"      CI at {a.sim_datetime}: "
          f"{min(ci.values()):.0f}-{max(ci.values()):.0f} gCO2/kWh   PUE {pue_str}")
    print(f"  endpoints: {ep_path}")
    print(f"  logs:      {log_dir}/")

    if a.foreground:
        signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))
        print("  attached; Ctrl-C to stop the fleet")
        try:
            while True:
                dead = [d for d, p in procs.items() if p.poll() is not None]
                if dead:
                    print(f"  {len(dead)} datacentre(s) died: "
                          f"{', '.join(dead[:8])}", file=sys.stderr)
                    shutdown()
                    sys.exit(1)
                time.sleep(1.0)
        except KeyboardInterrupt:
            shutdown()
    else:
        pids = " ".join(str(p.pid) for p in procs.values())
        print(f"  stop with: kill {pids}")


if __name__ == "__main__":
    main()

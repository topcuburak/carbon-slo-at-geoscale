#!/usr/bin/env python3
"""Collect energy/carbon stats from all running DC servers via FinalizeEnergy + PollStats.

Usage:
    python scripts/sweep/collect_energy.py [output_json] [--requests-csv PATH]

1. Reads the loadgen output CSV to compute sim_duration_s (max recv - min send).
2. Calls FinalizeEnergy(sim_duration_s) on every DC so each computes energy
   from its GPU time series over the correct simulation window.
3. Polls PollStats to collect energy_j, carbon_intensity, and carbon_g.

Output: JSON with per-DC and total energy/carbon.
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta

# Add project root to path so imports work when called from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import grpc
import inference_pb2
import inference_pb2_grpc

from common.pue import get_pue, pue_table


def compute_sim_duration(requests_csv: str) -> float:
    """Compute simulation duration from loadgen output CSV.

    Returns max(recv_time) - min(send_time) in seconds.
    send_time and recv_time are in %H:%M:%S.%f format (time-only, no date).

    Handles midnight crossing: if timestamps span both late evening (>=21h)
    and early morning (<6h), post-midnight times are shifted +24h so that
    min/max comparisons produce correct results.
    """
    send_times = []
    recv_times = []
    with open(requests_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            send_times.append(datetime.strptime(row["send_time"], "%H:%M:%S.%f"))
            recv_times.append(datetime.strptime(row["recv_time"], "%H:%M:%S.%f"))

    if not send_times:
        print("  WARNING: no OK requests in CSV, using 0s duration", file=sys.stderr)
        return 0.0

    # Detect midnight crossing: timestamps in both late evening and early morning
    all_times = send_times + recv_times
    has_late = any(t.hour >= 21 for t in all_times)
    has_early = any(t.hour < 6 for t in all_times)

    if has_late and has_early:
        DAY = timedelta(days=1)
        send_times = [t + DAY if t.hour < 12 else t for t in send_times]
        recv_times = [t + DAY if t.hour < 12 else t for t in recv_times]

    min_send = min(send_times)
    max_recv = max(recv_times)
    dt = (max_recv - min_send).total_seconds()
    return max(dt, 0.0)


def main():
    # Parse args: collect_energy.py <output_json> [--requests-csv PATH]
    output_path = "/dev/stdout"
    requests_csv = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--requests-csv" and i + 1 < len(args):
            requests_csv = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            output_path = args[i]
            i += 1
        else:
            i += 1

    # The map the four launchers merge into. It used to default to
    # scripts/local_endpoints.json, a hand-written 64-site fleet on base port
    # 50100 last touched in March -- so a manual run collected nothing and
    # reported it as a wall of connection refusals. The sweep sets
    # STATIC_ENDPOINTS_FILE to its own generated map and is unaffected.
    endpoints_file = os.environ.get("STATIC_ENDPOINTS_FILE",
                                    "results/logs/endpoints.json")
    if not os.path.isfile(endpoints_file):
        sys.exit(f"no endpoint map at {endpoints_file}. Start the fleet "
                 f"(datacenter/start.sh) or set STATIC_ENDPOINTS_FILE.")
    with open(endpoints_file) as f:
        all_endpoints = json.load(f)

    # Filter to DC endpoints (not scheduler or provisioner)
    dc_endpoints = {
        k: v for k, v in all_endpoints.items()
        if not k.startswith("sched-") and not k.startswith("provisioner")
    }

    # Step 1: Compute sim_duration
    sim_duration_s = float(os.environ.get("SIM_DURATION_S", "0"))
    if sim_duration_s <= 0 and requests_csv and os.path.isfile(requests_csv):
        sim_duration_s = compute_sim_duration(requests_csv)
        print(f"  Sim duration from CSV: {sim_duration_s:.3f}s", file=sys.stderr)
    elif sim_duration_s > 0:
        print(f"  Sim duration from env: {sim_duration_s:.1f}s", file=sys.stderr)
    else:
        print("  WARNING: no sim duration available, skipping FinalizeEnergy", file=sys.stderr)

    # Step 2: Call FinalizeEnergy on all DCs
    if sim_duration_s > 0:
        for dc_id, addr in sorted(dc_endpoints.items()):
            try:
                channel = grpc.insecure_channel(addr)
                stub = inference_pb2_grpc.DatacenterServiceStub(channel)
                stub.FinalizeEnergy(
                    inference_pb2.FinalizeEnergyRequest(sim_duration_s=sim_duration_s),
                    timeout=10.0,
                )
                channel.close()
            except Exception as e:
                print(f"  WARNING: FinalizeEnergy failed for {dc_id}: {e}", file=sys.stderr)

    # Step 3: Collect stats via PollStats
    results = {}
    total_energy_j = 0.0
    total_carbon_g = 0.0
    total_idle_j = 0.0
    total_host_j = 0.0
    total_work_carbon_g = 0.0

    for dc_id, addr in sorted(dc_endpoints.items()):
        try:
            channel = grpc.insecure_channel(addr)
            stub = inference_pb2_grpc.DatacenterServiceStub(channel)
            resp = stub.PollStats(
                inference_pb2.PollStatsRequest(src="energy_collector", epoch=0),
                timeout=5.0,
            )
            energy_j = resp.total_energy_j
            ci = resp.carbon_intensity
            # THIS datacentre's PUE, from the same table the datacentre server
            # read at startup. It used to be the global constant here while the
            # datacentre used its own, which agreed only for as long as every
            # site shared one value -- the moment they differ, this script and
            # the site it is reporting on disagree about the same joules.
            site_pue = get_pue(dc_id)
            carbon_g = energy_j * site_pue * ci / 3_600_000.0
            # The datacentre computed the same quantity from its own PUE and
            # its own energy. If the two part company, one of them is reading a
            # PUE the other is not -- an environment override on the site, or a
            # table this script cannot see -- and the carbon column is not
            # trustworthy. Say so rather than silently preferring one.
            reported = resp.carbon_emissions_g
            if reported > 0 and abs(reported - carbon_g) > max(1e-6, 1e-3 * reported):
                # Back out what the site must have used, and report THAT as
                # this site's PUE. Deferring to the site's carbon while still
                # printing the table's number would leave the summary line
                # describing a different calculation from the one above it --
                # the same reports-one-thing-computes-another this check exists
                # to catch.
                implied = (reported / carbon_g) * site_pue if carbon_g > 0 else site_pue
                print(f"  WARNING: {dc_id} reports {reported:.3f} gCO2 but this "
                      f"script computes {carbon_g:.3f} from PUE={site_pue:.3f}; "
                      f"the site is using PUE={implied:.3f}. Using the site's, "
                      f"and reporting its PUE.", file=sys.stderr)
                carbon_g = reported
                work_pue = implied
                site_pue = implied
            # The same joules split by what caused them. work is the part a
            # placement decision moves; idle and host are what the site costs
            # to have powered. Reported separately rather than chosen between,
            # because whether a policy comparison should carry the floor
            # depends on whether the site would have been up anyway -- a
            # question about the deployment, not about the simulator.
            idle_j = resp.idle_energy_j
            host_j = resp.host_energy_j
            work_j = max(0.0, energy_j - idle_j - host_j)
            work_carbon_g = work_j * site_pue * ci / 3_600_000.0

            results[dc_id] = {
                "energy_j": energy_j,
                "idle_energy_j": idle_j,
                "host_energy_j": host_j,
                "work_energy_j": work_j,
                "carbon_intensity": ci,
                "pue": site_pue,
                "carbon_g": carbon_g,
                "work_carbon_g": work_carbon_g,
            }
            total_energy_j += energy_j
            total_carbon_g += carbon_g
            total_idle_j += idle_j
            total_host_j += host_j
            total_work_carbon_g += work_carbon_g
            channel.close()
        except Exception as e:
            results[dc_id] = {"error": str(e)}

    total_work_j = max(0.0, total_energy_j - total_idle_j - total_host_j)
    # One number here would be a lie the moment two sites differ, so record
    # what was actually used per site alongside the range.
    used = sorted({round(r["pue"], 6) for r in results.values() if "pue" in r})
    output = {
        "pue_by_dc": {k: r["pue"] for k, r in results.items() if "pue" in r},
        "pue_uniform": (used[0] if len(used) == 1 else None),
        "pue_range": [used[0], used[-1]] if used else None,
        "sim_duration_s": sim_duration_s,
        "per_dc": results,
        "total_energy_j": total_energy_j,
        "total_energy_kwh": total_energy_j / 3_600_000.0,
        "total_carbon_g": total_carbon_g,
        # Two accounting views of the same run, so a figure can use either
        # without a second sweep. "total" charges the fleet for being powered;
        # "work" charges only what serving cost.
        "total_idle_energy_j": total_idle_j,
        "total_host_energy_j": total_host_j,
        "total_work_energy_j": total_work_j,
        "total_work_carbon_g": total_work_carbon_g,
        "floor_fraction": (
            (total_idle_j + total_host_j) / total_energy_j
            if total_energy_j > 0 else 0.0
        ),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print per-DC and total summary to stderr for sweep script
    for dc_id, info in sorted(results.items()):
        if "error" in info:
            print(f"    {dc_id:>20}  ERROR: {info['error']}", file=sys.stderr)
        else:
            print(f"    {dc_id:>20}  CI={info['carbon_intensity']:6.1f} gCO2/kWh"
                  f"  energy={info['energy_j']:12.0f}J"
                  f"  work={info['work_energy_j']:11.0f}J"
                  f"  carbon={info['carbon_g']:8.3f}g"
                  f"  work_carbon={info['work_carbon_g']:8.3f}g",
                  file=sys.stderr)
    if not used:
        sys.exit(f"no datacentre answered out of {len(dc_endpoints)} in "
                 f"{endpoints_file}. Energy was NOT collected; the errors "
                 f"above say why. Nothing was written to {output_path}.")
    pue_str = (f"{used[0]:.3f}" if len(used) == 1
               else f"{used[0]:.3f}-{used[-1]:.3f} over {len(used)} values")
    print(f"  TOTAL  energy={total_energy_j:.0f}J ({total_energy_j/3_600_000:.4f}kWh)  "
          f"Carbon: {total_carbon_g:.1f}g  PUE={pue_str}  sim={sim_duration_s:.1f}s",
          file=sys.stderr)
    print(f"  SPLIT  work={total_work_j:.0f}J  idle={total_idle_j:.0f}J  "
          f"host={total_host_j:.0f}J   floor={output['floor_fraction']*100:.0f}% of the total",
          file=sys.stderr)
    print(f"  WORK-ONLY carbon: {total_work_carbon_g:.1f}g  "
          f"({total_work_carbon_g/total_carbon_g*100:.0f}% of the reported total)"
          if total_carbon_g > 0 else "", file=sys.stderr)


if __name__ == "__main__":
    main()

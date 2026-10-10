#!/usr/bin/env bash
# =============================================================================
#  Smoke test for the parallel sweep config
# =============================================================================
#
# Runs Std ILP at rate=100 with the new 32-DC topology (4 zones × 8 DCs)
# to verify:
#   1. Engine starts cleanly at the larger DC count
#   2. ILP scheduler overhead (sched_p90) stays below the budget threshold
#
# A sched_p90 above ~400 ms means ILP solve time is too high for the new
# topology; either reduce DCs/zone or accept a wider TTFT budget. A
# sched_p90 below ~300 ms is healthy.
#
# Usage:
#   bash scripts/test/smoke_test_topology.sh
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="${ROOT:-results/smoke_topology_$(date +%Y%m%d_%H%M)}"
SEED="${SEED:-42}"
N_DCS_PER="${N_DCS_PER:-8}"
N_NODES_PER_DC="${N_NODES_PER_DC:-10}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  Smoke test: Std ILP at rate=100 on $((N_DCS_PER * 4)) DCs"
echo "  Output: $ROOT"
echo "============================================================"

# Generate topology
PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
    --seed "$SEED" \
    --n-dcs-per "$N_DCS_PER" \
    --n-nodes-per-dc "$N_NODES_PER_DC" \
    --output "$ROOT/topology.json" 2>&1 | head -10

# Run ONLY std ILP at ONLY rate=100, short duration
TTFT_BUDGET=2000 \
TPOT_BUDGET=50 \
BS_CAP=128 \
NUM_NODES="$N_NODES_PER_DC" \
TP_SIZE=2 \
POLL_MODE=async \
POLL_PERIOD_S=1 \
POLL_WAVE_COUNT=4 \
POLL_WAVE_INTERVAL_S=0.25 \
DURATION=60 \
RESULTS_ROOT="$ROOT" \
TOPOLOGY_JSON="$ROOT/topology.json" \
ARRIVAL_RATES="100" \
POLICIES="ilp_carbon_sla" \
bash scripts/sweep/sweep_all_policies.sh

echo
echo "============================================================"
echo "  Smoke test complete. Computing sched_p90..."
echo "============================================================"

# Extract sched_p90 from results.csv
RESULTS_CSV="$ROOT/ilp_carbon_sla/rate_100/results.csv"
if [ ! -f "$RESULTS_CSV" ]; then
    echo "ERROR: results.csv not found at $RESULTS_CSV"
    exit 1
fi

python3 - <<PYEOF
import pandas as pd
df = pd.read_csv("$RESULTS_CSV")
errs = df.error.notna().sum() if 'error' in df.columns else 0
print(f"  Total requests: {len(df)}")
print(f"  Errors:         {errs}")
print(f"  sched_ms p50:   {df.scheduling_ms.median():.2f}")
print(f"  sched_ms p90:   {df.scheduling_ms.quantile(0.9):.2f}")
print(f"  sched_ms p99:   {df.scheduling_ms.quantile(0.99):.2f}")
print(f"  ttft_ms  p50:   {df.ttft_ms.median():.2f}")
print(f"  ttft_ms  p90:   {df.ttft_ms.quantile(0.9):.2f}")
fwd_p90 = df.forwarding_ms.quantile(0.9)
print(f"  fwd_ms   p90:   {fwd_p90:.2f}")
user_ttft = df.scheduling_ms + df.forwarding_ms/2 + df.queue_ms + df.ttft_ms
print(f"  user_TTFT p50:  {user_ttft.median():.2f}")
print(f"  user_TTFT p90:  {user_ttft.quantile(0.9):.2f}")
print(f"  user_TTFT p99:  {user_ttft.quantile(0.99):.2f}")

s90 = df.scheduling_ms.quantile(0.9)
print()
if s90 < 300:
    print(f"  ✓ sched_p90 = {s90:.0f} ms — healthy. Proceed with full sweep.")
elif s90 < 400:
    print(f"  ⚠ sched_p90 = {s90:.0f} ms — borderline. Full sweep may be unfair to ILPs.")
else:
    print(f"  ✗ sched_p90 = {s90:.0f} ms — TOO HIGH. Reduce N_DCS_PER or widen TTFT_BUDGET.")
PYEOF

#!/usr/bin/env bash
# =============================================================================
#  Temporal sensitivity sweep — parallel orchestrator
# =============================================================================
#
# Runs sweep_parallel.sh (7 policies in parallel within rate) at multiple
# wall-clock timestamps to study how policy ranking depends on time of day.
# Topology is held fixed across hours so the only varying input is the
# carbon-intensity snapshot.
#
# Hours run sequentially; policies run in parallel within each hour.
# Estimated wallclock: ~30 min × 8 hours = ~4 hours.
#
# Usage:
#   bash scripts/sweep/sweep_temporal_parallel.sh
#
# Environment overrides:
#   HOURS="0 3 6 9 12 15 18 21"
#   SIM_DATE="2024-07-15"
#   SEED=42
#   ARRIVAL_RATES="200"        single rate keeps it tractable
#   N_DCS_PER=8 N_NODES_PER_DC=10
#   TTFT_BUDGET=2000 TPOT_BUDGET=50 BS_CAP=128
#   ROOT=results/temporal_parallel_<timestamp>
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

HOURS="${HOURS:-0 3 6 9 12 15 18 21}"
SIM_DATE="${SIM_DATE:-2024-07-15}"
SEED="${SEED:-42}"
ARRIVAL_RATES="${ARRIVAL_RATES:-200}"
ROOT="${ROOT:-results/temporal_parallel_$(date +%Y%m%d_%H%M)}"

# Pass-through to sweep_parallel.sh
export N_DCS_PER="${N_DCS_PER:-8}"
export N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
export TTFT_BUDGET="${TTFT_BUDGET:-2000}"
export TPOT_BUDGET="${TPOT_BUDGET:-50}"
export BS_CAP="${BS_CAP:-128}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  Temporal parallel sweep"
echo "  Date:           $SIM_DATE"
echo "  Hours:          $HOURS"
echo "  Seed:           $SEED"
echo "  Topology:       $N_DCS_PER DCs × $N_NODES_PER_DC nodes per zone × 4 zones"
echo "  Arrival rates:  $ARRIVAL_RATES"
echo "  TTFT budget:    ${TTFT_BUDGET} ms"
echo "  TPOT budget:    ${TPOT_BUDGET} ms"
echo "  BS_CAP:         $BS_CAP"
echo "  Output root:    $ROOT"
echo "============================================================"

# Pre-generate ONE topology shared across all hours
TOPO_FILE="$ROOT/topology.json"
PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
    --seed "$SEED" \
    --n-dcs-per "$N_DCS_PER" \
    --n-nodes-per-dc "$N_NODES_PER_DC" \
    --output "$TOPO_FILE" 2>&1 | head -10
echo

# Save outer-sweep manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "temporal_parallel",
  "date": "$SIM_DATE",
  "hours": [$(echo $HOURS | sed 's/ /, /g')],
  "seed": $SEED,
  "n_dcs_per": $N_DCS_PER,
  "n_nodes_per_dc": $N_NODES_PER_DC,
  "n_dcs_total": $((N_DCS_PER * 4)),
  "arrival_rates": "$ARRIVAL_RATES",
  "ttft_budget_ms": $TTFT_BUDGET,
  "tpot_budget_ms": $TPOT_BUDGET,
  "bs_cap": $BS_CAP,
  "started_at": "$(date -Iseconds)"
}
EOF

HOUR_IDX=0
for HOUR in $HOURS; do
    HOUR_PADDED=$(printf "%02d" "$HOUR")
    SIM_DATETIME="${SIM_DATE}T${HOUR_PADDED}:00:00"
    SUBDIR="$ROOT/hour_${HOUR_PADDED}"
    LOG="$ROOT/hour_${HOUR_PADDED}.log"

    # All hours reuse POLICY_SLOT_BASE=0. Hours run sequentially so
    # concurrency isolation is unnecessary; a 60s pause lets any
    # TIME_WAIT'd sockets clear. The earlier HOUR_IDX*8 scheme overflowed
    # the 16-bit TCP port limit at hour ≥ 2 (port = 50100 + slot*1000 >
    # 65535).
    export POLICY_SLOT_BASE=0
    if [ "$HOUR_IDX" -gt 0 ]; then
        echo "  (waiting 60s for TIME_WAIT to clear before next hour…)"
        sleep 60
    fi
    HOUR_IDX=$((HOUR_IDX + 1))

    echo
    echo "============================================================"
    echo "  → Hour ${HOUR_PADDED}:00 UTC  ($SIM_DATETIME)"
    echo "  → Slot base: $POLICY_SLOT_BASE"
    echo "  → Output:    $SUBDIR"
    echo "============================================================"
    t0=$(date +%s)

    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    ROOT="$SUBDIR" \
    POLICY_SLOT_BASE="$POLICY_SLOT_BASE" \
    bash scripts/sweep/sweep_parallel.sh > "$LOG" 2>&1

    dt=$(( $(date +%s) - t0 ))
    echo "  hour ${HOUR_PADDED} done in ${dt}s"
done

echo
echo "============================================================"
echo "  Temporal parallel sweep complete: $ROOT"
echo "============================================================"

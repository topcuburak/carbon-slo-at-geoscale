#!/usr/bin/env bash
# =============================================================================
#  Polling sensitivity sweep — parallel orchestrator
# =============================================================================
#
# Tests how the provisioner's poll mode + poll period affects each policy.
# Runs sync 1s + async at 0.5/1/2/4/8/16 second poll periods; each
# configuration runs sweep_parallel.sh (7 policies in parallel within rate)
# on a fixed topology.
#
# Configurations run sequentially with disjoint POLICY_SLOT_BASE ranges to
# avoid TIME_WAIT collisions. Estimated wallclock: ~14 min × 7 configs ≈
# 100 min at rate=2000.
#
# Usage:
#   bash scripts/sweep/sweep_polling_parallel.sh
#
# Environment overrides:
#   CONFIGS                     space-separated list (defaults below)
#   ARRIVAL_RATES="2000"        single rate keeps it tractable
#   N_DCS_PER=8 N_NODES_PER_DC=10
#   TTFT_BUDGET=2000 TPOT_BUDGET=50 BS_CAP=128
#   SEED=42 SIM_DATETIME="2024-07-15T12:00:00"
#   ROOT=results/sweeps/polling_parallel_<timestamp>
#
# Each config name encodes the mode and period: e.g. "sync_1s", "async_2s".
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="${ROOT:-results/sweeps/polling_parallel_$(date +%Y%m%d_%H%M)}"
SEED="${SEED:-42}"
ARRIVAL_RATES="${ARRIVAL_RATES:-2000}"
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"

# Configs: (name | mode | period_s)
# Format: <mode>_<period>s, with "p" replacing "." for fractional seconds
#   (e.g. async_0p5s = async mode, 0.5s period).
# Wave interval is auto-computed as period/POLL_WAVE_COUNT for async mode;
# in sync mode the wave parameters are unused.
# Default set: sync × {0.5,1,2,4,8,16}s + async × {0.5,1,2,4,8,16}s = 12 configs.
CONFIGS="${CONFIGS:-sync_0p5s sync_1s sync_2s sync_4s sync_8s sync_16s async_0p5s async_1s async_2s async_4s async_8s async_16s}"

# Pass-through to sweep_parallel.sh
export N_DCS_PER="${N_DCS_PER:-8}"
export N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
export TTFT_BUDGET="${TTFT_BUDGET:-2000}"
export TPOT_BUDGET="${TPOT_BUDGET:-50}"
export BS_CAP="${BS_CAP:-128}"
export POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  Polling parallel sweep"
echo "  Configs:        $CONFIGS"
echo "  Topology:       $N_DCS_PER DCs × $N_NODES_PER_DC nodes per zone × 4 zones"
echo "  Arrival rates:  $ARRIVAL_RATES"
echo "  TTFT budget:    ${TTFT_BUDGET} ms"
echo "  TPOT budget:    ${TPOT_BUDGET} ms"
echo "  BS_CAP:         $BS_CAP"
echo "  Output root:    $ROOT"
echo "============================================================"

# Pre-generate ONE topology shared across all configs
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
  "type": "polling_parallel",
  "configs": "$CONFIGS",
  "n_dcs_per": $N_DCS_PER,
  "n_nodes_per_dc": $N_NODES_PER_DC,
  "n_dcs_total": $((N_DCS_PER * 4)),
  "arrival_rates": "$ARRIVAL_RATES",
  "ttft_budget_ms": $TTFT_BUDGET,
  "tpot_budget_ms": $TPOT_BUDGET,
  "bs_cap": $BS_CAP,
  "datetime": "$SIM_DATETIME",
  "started_at": "$(date -Iseconds)"
}
EOF

# parse_config: extract mode (sync|async) and period (seconds, supporting "0p5" → 0.5)
parse_config() {
    local cfg="$1"
    local mode="${cfg%%_*}"      # sync_1s -> sync, async_0p5s -> async
    local period_token="${cfg#*_}"  # 1s, 0p5s
    period_token="${period_token%s}"      # strip trailing s
    period_token="${period_token//p/.}"   # 0p5 -> 0.5
    echo "$mode $period_token"
}

CFG_IDX=0
for CFG in $CONFIGS; do
    read -r MODE PERIOD <<< "$(parse_config "$CFG")"
    SUBDIR="$ROOT/$CFG"
    LOG="$ROOT/${CFG}.log"

    # Wave interval = period / POLL_WAVE_COUNT (async only)
    WAVE_INTERVAL=$(python3 -c "print($PERIOD / $POLL_WAVE_COUNT)")

    # All configs reuse POLICY_SLOT_BASE=0 — sequential execution makes
    # concurrency isolation unnecessary. The previous CFG_IDX*8 scheme
    # overflowed the 16-bit TCP port limit at config_idx ≥ 2 (port =
    # 50100 + slot*1000 > 65535).
    export POLICY_SLOT_BASE=0

    if [ "$CFG_IDX" -gt 0 ]; then
        echo "  (waiting 60s for TIME_WAIT to clear before next config…)"
        sleep 60
    fi
    CFG_IDX=$((CFG_IDX + 1))

    echo
    echo "============================================================"
    echo "  Config: $CFG  (mode=$MODE  period=${PERIOD}s  wave=${WAVE_INTERVAL}s)"
    echo "  Slot base:  $POLICY_SLOT_BASE  (ports $((50100 + POLICY_SLOT_BASE * 1000))+)"
    echo "  Output:     $SUBDIR"
    echo "============================================================"
    t0=$(date +%s)

    POLL_MODE="$MODE" \
    POLL_PERIOD_S="$PERIOD" \
    POLL_WAVE_INTERVAL_S="$WAVE_INTERVAL" \
    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    ROOT="$SUBDIR" \
    POLICY_SLOT_BASE="$POLICY_SLOT_BASE" \
    bash scripts/sweep/sweep_parallel.sh > "$LOG" 2>&1

    dt=$(( $(date +%s) - t0 ))
    echo "  $CFG done in ${dt}s"
done

echo
echo "============================================================"
echo "  Polling parallel sweep complete: $ROOT"
echo "============================================================"

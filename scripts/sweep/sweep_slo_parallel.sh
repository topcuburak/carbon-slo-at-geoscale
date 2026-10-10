#!/usr/bin/env bash
# =============================================================================
#  SLO budget sensitivity sweep — parallel orchestrator
# =============================================================================
#
# Tests how scheduling-policy ranking changes as the user-facing TTFT/TPOT
# budgets are tightened or loosened. Sweeps the TTFT axis (with TPOT fixed
# at 50ms) and the TPOT axis (with TTFT fixed at 2000ms) — the joint
# baseline (TTFT=2000, TPOT=50) appears once, so the default has 9 unique
# configs rather than the 5×5 = 25 of a full grid.
#
# Default TTFT axis: 1000, 1500, 2000, 3000, 5000 ms
# Default TPOT axis: 30, 40, 50, 75, 100 ms
#
# Configurations run sequentially with disjoint POLICY_SLOT_BASE ranges so
# TIME_WAIT'd ports from one config don't trample the next. Each config
# launches sweep_parallel.sh (7 policies in parallel within rate).
#
# Usage:
#   bash scripts/sweep/sweep_slo_parallel.sh
#
# Environment overrides:
#   CONFIGS                       space-separated; format ttft<X>_tpot<Y>
#                                 (numeric milliseconds, no decimals)
#   ARRIVAL_RATES="2000"          single rate keeps it tractable
#   N_DCS_PER=8 N_NODES_PER_DC=10
#   BS_CAP=128
#   SEED=42 SIM_DATETIME="2024-07-15T12:00:00"
#   ROOT=results/sweeps/slo_parallel_<timestamp>
#
# Each config name encodes both budgets: "ttft1000_tpot50", "ttft2000_tpot30".
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="${ROOT:-results/sweeps/slo_parallel_$(date +%Y%m%d_%H%M)}"
SEED="${SEED:-42}"
ARRIVAL_RATES="${ARRIVAL_RATES:-2000}"
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"

# Default 9 configs: 5 along TTFT axis (TPOT=50 fixed) + 4 along TPOT axis
# (TTFT=2000 fixed; baseline ttft2000_tpot50 already in TTFT axis).
CONFIGS="${CONFIGS:-\
ttft1000_tpot50 ttft1500_tpot50 ttft2000_tpot50 ttft3000_tpot50 ttft5000_tpot50 \
ttft2000_tpot30 ttft2000_tpot40 ttft2000_tpot75 ttft2000_tpot100}"

# Pass-through to sweep_parallel.sh
export N_DCS_PER="${N_DCS_PER:-8}"
export N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
export BS_CAP="${BS_CAP:-128}"
export POLL_MODE="${POLL_MODE:-async}"
export POLL_PERIOD_S="${POLL_PERIOD_S:-1}"
export POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}"
export POLL_WAVE_INTERVAL_S="${POLL_WAVE_INTERVAL_S:-0.25}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  SLO budget parallel sweep"
echo "  Configs:        $CONFIGS"
echo "  Topology:       $N_DCS_PER DCs × $N_NODES_PER_DC nodes per zone × 4 zones"
echo "  Arrival rates:  $ARRIVAL_RATES"
echo "  BS_CAP:         $BS_CAP"
echo "  Output root:    $ROOT"
echo "============================================================"

# Pre-generate ONE topology shared across all configs so SLO is the only
# thing varying.
TOPO_FILE="$ROOT/topology.json"
PYTHONPATH=proto:. python3 scripts/sweep/generate_experiment_config.py \
    --seed "$SEED" \
    --n-dcs-per "$N_DCS_PER" \
    --n-nodes-per-dc "$N_NODES_PER_DC" \
    --output "$TOPO_FILE" 2>&1 | head -10
echo

# Save outer-sweep manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "slo_parallel",
  "configs": "$CONFIGS",
  "n_dcs_per": $N_DCS_PER,
  "n_nodes_per_dc": $N_NODES_PER_DC,
  "n_dcs_total": $((N_DCS_PER * 4)),
  "arrival_rates": "$ARRIVAL_RATES",
  "bs_cap": $BS_CAP,
  "datetime": "$SIM_DATETIME",
  "started_at": "$(date -Iseconds)"
}
EOF

# parse_config: extract TTFT and TPOT budgets from the config name.
parse_config() {
    local cfg="$1"
    local ttft_part="${cfg%%_*}"   # ttft1000
    local tpot_part="${cfg#*_}"    # tpot50
    local ttft="${ttft_part#ttft}"
    local tpot="${tpot_part#tpot}"
    echo "$ttft $tpot"
}

CFG_IDX=0
for CFG in $CONFIGS; do
    read -r TTFT TPOT <<< "$(parse_config "$CFG")"
    SUBDIR="$ROOT/$CFG"
    LOG="$ROOT/${CFG}.log"

    # All configs reuse POLICY_SLOT_BASE=0 (slot range 0-6, ports
    # 50100-56300). Configs run sequentially so concurrency isolation is
    # unnecessary; we just need to wait briefly between configs for any
    # TIME_WAIT'd sockets from the previous config to clear. The earlier
    # CFG_IDX*8 scheme pushed port = 50100 + slot*1000 beyond the 16-bit
    # 65535 limit at config_idx ≥ 2, causing gRPC bind failures.
    export POLICY_SLOT_BASE=0

    if [ "$CFG_IDX" -gt 0 ]; then
        echo "  (waiting 60s for TIME_WAIT to clear before next config…)"
        sleep 60
    fi
    CFG_IDX=$((CFG_IDX + 1))

    echo
    echo "============================================================"
    echo "  Config: $CFG  (TTFT=${TTFT}ms  TPOT=${TPOT}ms)"
    echo "  Slot base:  $POLICY_SLOT_BASE  (ports $((50100 + POLICY_SLOT_BASE * 1000))+)"
    echo "  Output:     $SUBDIR"
    echo "============================================================"
    t0=$(date +%s)

    TTFT_BUDGET="$TTFT" \
    TPOT_BUDGET="$TPOT" \
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
echo "  SLO budget parallel sweep complete: $ROOT"
echo "============================================================"

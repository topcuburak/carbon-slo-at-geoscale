#!/usr/bin/env bash
# =============================================================================
#  Parallel sweep orchestrator
# =============================================================================
#
# Fans out one sweep_all_policies.sh sub-invocation per (policy) at a given
# arrival rate, with each sub-instance using a unique POLICY_SLOT to offset
# its gRPC ports. All policies for one rate run in parallel; rates run
# sequentially.
#
# Wallclock target: ~3 min per rate × N rates (vs ~3 × n_policies sequential).
#
# Usage:
#   bash scripts/sweep/sweep_parallel.sh
#
# Environment overrides (defaults shown):
#   ROOT="results/sweep_parallel_$(date +%Y%m%d_%H%M)"
#   ARRIVAL_RATES="50 100 200 500 1000 2000 5000"
#   POLICIES="round_robin least_loaded carbon_aware:0.25 carbon_aware:0.5 \
#             carbon_aware:0.9 ilp_carbon_sla iterative_ilp_carbon_sla"
#   N_DCS_PER=8 N_NODES_PER_DC=10 N_SCHEDULERS=4 SEED=42
#   TTFT_BUDGET=2000 TPOT_BUDGET=50 BS_CAP=128
#   POLL_MODE=async POLL_PERIOD_S=1 POLL_WAVE_COUNT=4 POLL_WAVE_INTERVAL_S=0.25
#   SIM_DATETIME="2024-07-15T12:00:00"
#   DURATION=120
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

# ── Configuration (all env-overridable) ──
ROOT="${ROOT:-results/sweep_parallel_$(date +%Y%m%d_%H%M)}"
SEED="${SEED:-42}"
DURATION="${DURATION:-120}"
N_DCS_PER="${N_DCS_PER:-8}"
N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"

ARRIVAL_RATES="${ARRIVAL_RATES:-50 100 200 500 1000 2000 5000}"

# Policy list — each element runs in its own parallel slot
DEFAULT_POLICIES=(
    "round_robin"
    "least_loaded"
    "carbon_aware:0.25"
    "carbon_aware:0.5"
    "carbon_aware:0.9"
    "ilp_carbon_sla"
    "iterative_ilp_carbon_sla"
)
if [ -n "${POLICIES:-}" ]; then
    # shellcheck disable=SC2206
    POLICY_LIST=( $POLICIES )
else
    POLICY_LIST=( "${DEFAULT_POLICIES[@]}" )
fi

# SLO + capacity (passed through to sweep_all_policies.sh)
export TTFT_BUDGET="${TTFT_BUDGET:-2000}"
export TPOT_BUDGET="${TPOT_BUDGET:-50}"
export BS_CAP="${BS_CAP:-128}"
# Declared beside BS_CAP because the two are coupled -- BS_CAP is queue
# headroom counted in semaphore-cycles of this -- and because the scheduler
# now reads it too, to gate the admission-wait term in its TTFT estimate.
# It would otherwise reach the schedulers only through
# sweep_all_policies.sh's own default, which is easy to change without
# noticing that a policy decision depends on it.
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
export NUM_NODES="${N_NODES_PER_DC}"
export TP_SIZE="${TP_SIZE:-2}"
export POLL_MODE="${POLL_MODE:-async}"
export POLL_PERIOD_S="${POLL_PERIOD_S:-1}"
export POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}"
export POLL_WAVE_INTERVAL_S="${POLL_WAVE_INTERVAL_S:-0.25}"
export DURATION SIM_DATETIME SEED

mkdir -p "$ROOT"

echo "============================================================"
echo "  Parallel sweep orchestrator"
echo "  Output:           $ROOT"
echo "  Topology:         $N_DCS_PER DCs × $N_NODES_PER_DC nodes per zone × 4 zones"
echo "  Total DCs:        $((N_DCS_PER * 4))"
echo "  Arrival rates:    $ARRIVAL_RATES"
echo "  Policies (slots): ${#POLICY_LIST[@]} parallel"
for i in "${!POLICY_LIST[@]}"; do
    echo "    slot $i: ${POLICY_LIST[$i]}"
done
echo "  TTFT budget:      ${TTFT_BUDGET} ms"
echo "  TPOT budget:      ${TPOT_BUDGET} ms"
echo "  BS_CAP:           $BS_CAP"
echo "  Poll:             $POLL_MODE / period=${POLL_PERIOD_S}s"
echo "  Datetime:         $SIM_DATETIME"
echo "============================================================"

# ── Resolve topology ──
# If TOPOLOGY_JSON is provided (e.g. from sweep_temporal.sh / sweep_spatial.sh
# pre-generating per-slice topologies), reuse it; otherwise generate fresh.
TOPO_FILE="$ROOT/topology.json"
if [ -n "${TOPOLOGY_JSON:-}" ] && [ -f "$TOPOLOGY_JSON" ]; then
    echo "Using pre-built topology: $TOPOLOGY_JSON"
    SRC_REAL="$(realpath "$TOPOLOGY_JSON")"
    DST_REAL="$(realpath -m "$TOPO_FILE")"
    if [ "$SRC_REAL" != "$DST_REAL" ]; then
        cp "$TOPOLOGY_JSON" "$TOPO_FILE"
    fi
else
    echo "Generating topology ($N_DCS_PER DCs × 4 zones, seed=$SEED)..."
    PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
        --seed "$SEED" \
        --n-dcs-per "$N_DCS_PER" \
        --n-nodes-per-dc "$N_NODES_PER_DC" \
        --output "$TOPO_FILE" 2>&1 | head -10
fi
echo

# ── Save manifest ──
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "parallel_sweep",
  "n_dcs_per": $N_DCS_PER,
  "n_nodes_per_dc": $N_NODES_PER_DC,
  "n_schedulers": 4,
  "n_dcs_total": $((N_DCS_PER * 4)),
  "arrival_rates": "$ARRIVAL_RATES",
  "policies": [$(printf '"%s",' "${POLICY_LIST[@]}" | sed 's/,$//')],
  "ttft_budget_ms": $TTFT_BUDGET,
  "tpot_budget_ms": $TPOT_BUDGET,
  "bs_cap": $BS_CAP,
  "poll_mode": "$POLL_MODE",
  "poll_period_s": "$POLL_PERIOD_S",
  "datetime": "$SIM_DATETIME",
  "duration_s": $DURATION,
  "started_at": "$(date -Iseconds)"
}
EOF

# ── For each rate, fan out all policies in parallel ──
for RATE in $ARRIVAL_RATES; do
    echo
    echo "============================================================"
    echo "  Rate $RATE req/s — launching ${#POLICY_LIST[@]} policies in parallel"
    echo "============================================================"
    rate_t0=$(date +%s)

    PIDS=()
    SLOT_BASE="${POLICY_SLOT_BASE:-0}"
    for SLOT in "${!POLICY_LIST[@]}"; do
        POLICY_SPEC="${POLICY_LIST[$SLOT]}"
        # Effective slot includes the outer-loop offset (e.g. spatial/temporal
        # wrappers set POLICY_SLOT_BASE to give each experiment its own range)
        EFFECTIVE_SLOT=$((SLOT_BASE + SLOT))
        SUB_RESULTS="$ROOT"
        SUB_LOG="$ROOT/policy_slot${SLOT}_rate${RATE}.log"

        echo "  → slot=$SLOT (effective=$EFFECTIVE_SLOT)  policy=$POLICY_SPEC  ports=$((50100+EFFECTIVE_SLOT*1000))-$((50300+EFFECTIVE_SLOT*1000))  log=$SUB_LOG"

        POLICY_SLOT="$EFFECTIVE_SLOT" \
        RESULTS_ROOT="$SUB_RESULTS" \
        TOPOLOGY_JSON="$TOPO_FILE" \
        ARRIVAL_RATES="$RATE" \
        POLICIES="$POLICY_SPEC" \
        bash scripts/sweep/sweep_all_policies.sh > "$SUB_LOG" 2>&1 &
        PIDS+=($!)
    done

    # Wait for all parallel instances to finish
    fail=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || fail=$((fail + 1))
    done

    rate_dt=$(( $(date +%s) - rate_t0 ))
    if [ "$fail" -gt 0 ]; then
        echo "  ⚠ $fail of ${#PIDS[@]} parallel runs at rate=$RATE failed (check logs)"
    fi
    echo "  done in ${rate_dt}s"
done

echo
echo "============================================================"
echo "  Parallel sweep complete: $ROOT"
echo "============================================================"
echo
echo "Run analyze_sweep.py to produce summary.csv:"
echo "  python scripts/sweep/analyze_sweep.py $ROOT"

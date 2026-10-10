#!/usr/bin/env bash
# Quick test: 2 groups × 2 DCs, two policies (round_robin, carbon_aware)
# Tests loadgen verbosity=2 output.
#
# Usage: bash scripts/test/test_verbosity.sh

set -e
cd "$(dirname "$0")/../.."

DURATION=10
ARRIVAL_RATE=5
DCS_PER_GROUP=2
ALPHA=0.5
LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv}"
NUM_NODES="${NUM_NODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
POLL_PERIOD=2
LOGDIR="/tmp/grpc_verbosity_test"

mkdir -p "$LOGDIR"

# Kill any previous run
pkill -f "python -m datacenter.server" 2>/dev/null || true
pkill -f "python -m scheduler.server" 2>/dev/null || true
pkill -f "python -m provisioner.server" 2>/dev/null || true
sleep 1

# ── Carbon intensities ────────────────────────────────────────────
SIM_DATETIME="${SIM_DATETIME:-2024-10-15T06}"
CARBON_DATA_ROOT="${CARBON_DATA_ROOT:-carbon_intensity}"
declare -A CI
eval "$(CARBON_DATA_ROOT="$CARBON_DATA_ROOT" SIM_YEAR="${SIM_YEAR:-2024}" \
    python3 scripts/sweep/generate_ci.py "$SIM_DATETIME")"

# ── Only 2 groups, 2 DCs each ────────────────────────────────────
SCHED_GROUPS="europe_north us_east"
ALL_DCS="DK_East DK_West US_ISONE US_NYISO"
NUM_DCS=4

declare -A DC_PORTS
declare -A SCHED_PORTS
declare -A DC_GROUP
DC_GROUP[DK_East]=europe_north
DC_GROUP[DK_West]=europe_north
DC_GROUP[US_ISONE]=us_east
DC_GROUP[US_NYISO]=us_east

DC_PORT=50100
SCHED_PORT=50200

for GROUP in $SCHED_GROUPS; do
    SCHED_PORTS[$GROUP]=$SCHED_PORT
    SCHED_PORT=$((SCHED_PORT + 1))
done

for DC in $ALL_DCS; do
    DC_PORTS[$DC]=$DC_PORT
    DC_PORT=$((DC_PORT + 1))
done

# ── Generate endpoints.json ──────────────────────────────────────
ENDPOINTS="$LOGDIR/endpoints.json"
{
    echo "{"
    first=true
    for DC in $ALL_DCS; do
        $first || echo ","
        first=false
        printf '  "%s": "localhost:%d"' "$DC" "${DC_PORTS[$DC]}"
    done
    for GROUP in $SCHED_GROUPS; do
        echo ","
        printf '  "sched-%s": "localhost:%d"' "$GROUP" "${SCHED_PORTS[$GROUP]}"
    done
    echo ","
    printf '  "provisioner": "localhost:50300"'
    echo ""
    echo "}"
} > "$ENDPOINTS"

echo "=== Endpoints ==="
cat "$ENDPOINTS"
echo ""

start_infra() {
    local POLICY=$1
    local DC_IDX=0
    echo "=== Starting $NUM_DCS DCs (2 groups × 2 DCs) ==="
    for DC in $ALL_DCS; do
        DC_ID="$DC" \
        DC_INDEX="$DC_IDX" \
        GRPC_PORT="${DC_PORTS[$DC]}" \
        CARBON_INTENSITY="${CI[$DC]}" \
        LATENCY_CSV="$LATENCY_CSV" \
        NUM_NODES="$NUM_NODES" \
        TP_SIZE="$TP_SIZE" \
        VERBOSITY=2 \
        LOG_TAG="$POLICY" \
        STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
          python -m datacenter.server > "$LOGDIR/dc_${DC}.log" 2>&1 &
        echo "  DC$DC_IDX:$DC  port=${DC_PORTS[$DC]}  CI=${CI[$DC]}"
        DC_IDX=$((DC_IDX + 1))
    done

    echo "=== Starting 2 schedulers (policy=$POLICY, alpha=$ALPHA) ==="
    for GROUP in $SCHED_GROUPS; do
        SCHED_ID="sched-${GROUP}" \
        SCHEDULER_GROUP="$GROUP" \
        ROUTING_POLICY="$POLICY" \
        ROUTING_ALPHA="$ALPHA" \
        SIMULATE_RTT="true" \
        GRPC_PORT="${SCHED_PORTS[$GROUP]}" \
        VERBOSITY=2 \
        LOG_TAG="$POLICY" \
        STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
          python -m scheduler.server > "$LOGDIR/sched_${GROUP}.log" 2>&1 &
        echo "  Scheduler sched-${GROUP}  port=${SCHED_PORTS[$GROUP]}"
    done

    echo "=== Starting provisioner ==="
    POLL_PERIOD_S="$POLL_PERIOD" \
    GRPC_PORT=50300 \
    LATENCY_CSV="$LATENCY_CSV" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m provisioner.server > "$LOGDIR/provisioner.log" 2>&1 &
    echo "  Provisioner on port 50300"

    sleep 4
}

stop_infra() {
    pkill -f "python -m datacenter.server" 2>/dev/null || true
    pkill -f "python -m scheduler.server" 2>/dev/null || true
    pkill -f "python -m provisioner.server" 2>/dev/null || true
    sleep 2
}

run_loadgen() {
    local POLICY=$1
    local VERBOSITY=$2
    local OUT="$LOGDIR/results_${POLICY}_v${VERBOSITY}.csv"
    echo ""
    echo ">>> Loadgen: policy=$POLICY  verbosity=$VERBOSITY  duration=${DURATION}s  rate=${ARRIVAL_RATE}req/s"
    echo ">>> Output CSV: $OUT"
    echo ""
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m loadgen.client \
        --duration "$DURATION" \
        --arrival-distribution poisson \
        --arrival-rate "$ARRIVAL_RATE" \
        --concurrent 16 \
        --output "$OUT" \
        --verbosity "$VERBOSITY" \
        --tag "$POLICY"
    echo ""
}

# ── Test 1: round_robin with verbosity=2 ─────────────────────────
echo "============================================================"
echo "  TEST 1: round_robin  (verbosity=2)"
echo "============================================================"
start_infra round_robin
run_loadgen round_robin 2
stop_infra

# ── Test 2: carbon_aware with verbosity=2 ────────────────────────
echo "============================================================"
echo "  TEST 2: carbon_aware  alpha=$ALPHA  (verbosity=2)"
echo "============================================================"
start_infra carbon_aware
run_loadgen carbon_aware 2
stop_infra

echo ""
echo "=== Done. Results in $LOGDIR ==="
ls -la "$LOGDIR"/results_*.csv 2>/dev/null
echo ""
echo "To inspect per-request details:"
echo "  head -5 $LOGDIR/results_round_robin_v2.csv"
echo "  head -5 $LOGDIR/results_carbon_aware_v2.csv"

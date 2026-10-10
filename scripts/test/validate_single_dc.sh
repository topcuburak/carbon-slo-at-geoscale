#!/bin/bash
# Validate simulator latency accuracy using a single DC setup.
# Runs at multiple arrival rates to cover different batch size regimes.
#
# Usage:
#   LATENCY_CSV=datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv \
#   TP_SIZE=2 \
#   bash scripts/test/validate_single_dc.sh
#
# Results saved to results/validation/<model_tag>/

set -euo pipefail

LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv}"
TP_SIZE="${TP_SIZE:-2}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
NUM_NODES="${NUM_NODES:-1}"
DURATION="${DURATION:-30}"
MODEL_TAG="${MODEL_TAG:-default}"

# Single DC, single scheduler
DC_PORT=50100
SCHED_PORT=50200
PROV_PORT=50300
CI=100.0  # arbitrary, not relevant for latency validation

RESULTS_DIR="results/validation/${MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$(pwd)/proto:${PYTHONPATH:-}"

# Rates to test: low (bs~1), medium (bs~4-8), high (bs~16-32)
RATES=(1 2 5 10 20)

ENDPOINTS="$RESULTS_DIR/endpoints.json"

# Use a real DC name so REGION_TO_GROUP mapping works
DC_NAME="DK_East"
SCHED_GROUP="europe_north"
SCHED_NAME="sched-${SCHED_GROUP}"

# Generate minimal endpoints file (flat dict format)
cat > "$ENDPOINTS" <<EOJSON
{
  "${SCHED_NAME}": "localhost:${SCHED_PORT}",
  "${DC_NAME}": "localhost:${DC_PORT}",
  "provisioner": "localhost:${PROV_PORT}"
}
EOJSON

kill_all() {
    pkill -9 -f "python -m datacenter.server" 2>/dev/null || true
    pkill -9 -f "python -m scheduler.server" 2>/dev/null || true
    pkill -9 -f "python -m provisioner.server" 2>/dev/null || true
    pkill -9 -f "python -m loadgen.client" 2>/dev/null || true
    sleep 1
}

echo "================================================================"
echo "  Single-DC Latency Validation"
echo "  Latency CSV: $LATENCY_CSV"
echo "  TP size:     $TP_SIZE"
echo "  Max BS:      $MAX_BATCH_SIZE"
echo "  Duration:    ${DURATION}s per rate"
echo "  Rates:       ${RATES[*]}"
echo "  Results:     $RESULTS_DIR"
echo "================================================================"

for RATE in "${RATES[@]}"; do
    echo ""
    echo "--- Rate: $RATE req/s ---"
    kill_all

    # Start DC
    DC_ID="$DC_NAME" \
    GRPC_PORT="$DC_PORT" \
    CARBON_INTENSITY="$CI" \
    LATENCY_CSV="$LATENCY_CSV" \
    NUM_NODES="$NUM_NODES" \
    TP_SIZE="$TP_SIZE" \
    MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
    VERBOSITY=2 \
    python -m datacenter.server > "$RESULTS_DIR/dc_rate${RATE}.log" 2>&1 &
    sleep 2

    # Start scheduler
    SCHED_ID="$SCHED_NAME" \
    SCHEDULER_GROUP="$SCHED_GROUP" \
    ROUTING_POLICY="round_robin" \
    ROUTING_ALPHA="0" \
    SIMULATE_RTT="false" \
    GRPC_PORT="$SCHED_PORT" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
    python -m scheduler.server > "$RESULTS_DIR/sched_rate${RATE}.log" 2>&1 &
    sleep 1

    # Start provisioner
    POLL_PERIOD_S="2" \
    GRPC_PORT="$PROV_PORT" \
    LATENCY_CSV="$LATENCY_CSV" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
    python -m provisioner.server > "$RESULTS_DIR/prov_rate${RATE}.log" 2>&1 &
    sleep 2

    # Run loadgen
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
    python -m loadgen.client \
        --duration "$DURATION" \
        --arrival-distribution poisson \
        --arrival-rate "$RATE" \
        --concurrent 5000 \
        --trace-csv "$LATENCY_CSV" \
        --output "$RESULTS_DIR/loadgen_rate${RATE}.csv" \
        2>&1 | tee -a "$RESULTS_DIR/loadgen_rate${RATE}.log"

    echo "  Done rate=$RATE"
done

kill_all

echo ""
echo "================================================================"
echo "  Validation complete. Results in $RESULTS_DIR"
echo "================================================================"

#!/usr/bin/env bash
# Quick test: round_robin vs carbon_aware alpha=0.05
# Usage: bash scripts/test/quick_test.sh

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/proto:$PYTHONPATH"

DURATION="${1:-10}"
ARRIVAL_RATE="${2:-100}"
DCS_PER_GROUP="${3:-2}"
# Two different things that used to share one variable. LATENCY_CSV is the
# per-dataset lookup the datacentres and provisioner read; TRACE_CSV is the
# WORKLOAD the load generator replays. Passing the lookup as the workload
# happened to parse -- it carries in_tokens and out_tokens -- but it is not one
# of the scenario traces and carries no predicted_osl.
LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv}"
POLL_PERIOD="${POLL_PERIOD_S:-2}"
LOGDIR="results/logs/quick"
RESULTS_DIR="results/sweep/quick"
SUMMARY_CSV="results/quick_summary.csv"
NUM_NODES="${NUM_NODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
# The schedulers need this: any policy that estimates latency loads the
# analytical model from it, and without it carbon_aware failed 200 of 200
# requests here while the run still looked like it had produced results.
MODEL_NAME="${MODEL_NAME:-llama3_8B}"
# The short scenario for this model: mean ISL 841, so a smoke test finishes.
# Swap for mixed_ or long_ to exercise the long-context regime.
TRACE_CSV="${TRACE_CSV:-loadgen/scenarios/mlperf_${MODEL_NAME}.csv}"
SIM_DATETIME="${SIM_DATETIME:-2024-10-15T06}"
CARBON_DATA_ROOT="${CARBON_DATA_ROOT:-carbon_intensity}"

ENDPOINTS="$LOGDIR/endpoints.json"
mkdir -p "$LOGDIR" "$RESULTS_DIR" "$(dirname "$SUMMARY_CSV")"

# ── Carbon intensities ──
declare -A CI
eval "$(CARBON_DATA_ROOT="$CARBON_DATA_ROOT" SIM_YEAR="${SIM_YEAR:-2024}" \
    python3 scripts/sweep/generate_ci.py "$SIM_DATETIME")"
echo "  CI snapshot: $SIM_DATETIME"

# ── DC lists per group ──
EUROPE_NORTH=(DK_East DK_West Estonia Finland GreatBritain Ireland Latvia Lithuania Netherlands Norway SE_North SE_Central SE_South Belgium Poland)
EUROPE_SOUTH=(Austria Czechia France Germany Hungary IT_North IT_CentralNorth IT_CentralSouth IT_South IT_Sardinia IT_Sicily Portugal Serbia Slovakia Slovenia Spain Switzerland Croatia)
US_EAST=(US_ISONE US_NYISO US_PJM US_MISO US_DukeCarolinas US_DukeProgressE US_DukeProgressW US_SCEG)
US_SOUTH=(US_Southern US_Tallahassee US_FLMunicipal US_FPL US_Gainesville US_TVA US_ERCOT US_SPP)
US_WEST=(US_APS US_Avista US_BANC US_BPA US_CAISO US_ElPaso US_IdahoPower US_LADWP US_NWEnergy US_PacifiCorpE US_PGE US_PSCo US_Turlock US_WAPA US_AssocElec)
MIDDLE_EAST=(Israel Turkey Cyprus Greece Bulgaria Romania SouthAfrica)
ASIA=(AU_NSW AU_NT AU_QLD AU_VIC IN_East IN_NorthEast IN_North IN_South IN_West JP_Chubu JP_Chugoku JP_Hokkaido JP_Hokuriku JP_Kansai JP_Okinawa JP_Tohoku JP_Tokyo PH_Luzon PH_Mindanao PH_Visayas Singapore SouthKorea Taiwan NewZealand)
SOUTH_AMERICA=(BR_Central BR_North BR_NorthEast BR_South Chile Peru Nicaragua)

ALL_GROUPS="europe_north europe_south us_east us_south us_west middle_east asia south_america"

select_dcs() {
    local -n GROUP_ARR=$1
    local N=$2
    local MAX=${#GROUP_ARR[@]}
    local ACTUAL=$(( N < MAX ? N : MAX ))
    echo "${GROUP_ARR[@]:0:$ACTUAL}"
}

declare -A DC_PORTS
declare -A SCHED_PORTS
ALL_DCS=""
DC_PORT=50100
SCHED_PORT=50200

for GROUP in $ALL_GROUPS; do
    SCHED_PORTS[$GROUP]=$SCHED_PORT
    SCHED_PORT=$((SCHED_PORT + 1))
    GROUP_UPPER=$(echo "$GROUP" | tr '[:lower:]' '[:upper:]')
    SELECTED=$(select_dcs "$GROUP_UPPER" "$DCS_PER_GROUP")
    for DC in $SELECTED; do
        DC_PORTS[$DC]=$DC_PORT
        DC_PORT=$((DC_PORT + 1))
        ALL_DCS="$ALL_DCS $DC"
    done
done

ALL_DCS="${ALL_DCS# }"
NUM_DCS=$(echo $ALL_DCS | wc -w)
NUM_GROUPS=$(echo $ALL_GROUPS | wc -w)

generate_endpoints() {
    echo "{"
    local first=true
    for DC in $ALL_DCS; do
        $first || echo ","
        first=false
        printf '  "%s": "localhost:%d"' "$DC" "${DC_PORTS[$DC]}"
    done
    for GROUP in $ALL_GROUPS; do
        echo ","
        printf '  "sched-%s": "localhost:%d"' "$GROUP" "${SCHED_PORTS[$GROUP]}"
    done
    echo ","
    printf '  "provisioner": "localhost:50300"'
    echo ""
    echo "}"
}

generate_endpoints > "$ENDPOINTS"

kill_all() {
    pkill -9 -f 'python -m datacenter.server' 2>/dev/null || true
    pkill -9 -f 'python -m scheduler.server' 2>/dev/null || true
    pkill -9 -f 'python -m provisioner.server' 2>/dev/null || true
    sleep 2
}

start_dcs() {
    local LOG_TAG="$1"
    for DC in $ALL_DCS; do
        DC_ID="$DC" \
        GRPC_PORT="${DC_PORTS[$DC]}" \
        CARBON_INTENSITY="${CI[$DC]}" \
        LATENCY_CSV="$LATENCY_CSV" \
        NUM_NODES="$NUM_NODES" \
        TP_SIZE="$TP_SIZE" \
        STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
        VERBOSITY=0 \
          python -m datacenter.server > "$LOGDIR/${LOG_TAG}_dc_${DC}.log" 2>&1 &
    done
    echo "  Started $NUM_DCS datacenters"
}

start_schedulers() {
    local POLICY="$1"
    local ALPHA="$2"
    local LOG_TAG="$3"
    for GROUP in $ALL_GROUPS; do
        SCHED_ID="sched-${GROUP}" \
        SCHEDULER_GROUP="$GROUP" \
        ROUTING_POLICY="$POLICY" \
        ROUTING_ALPHA="$ALPHA" \
        SIMULATE_RTT="true" \
        GRPC_PORT="${SCHED_PORTS[$GROUP]}" \
        STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
        MODEL_NAME="$MODEL_NAME" \
        TP_SIZE="$TP_SIZE" \
          python -m scheduler.server > "$LOGDIR/${LOG_TAG}_sched_${GROUP}.log" 2>&1 &
    done
    echo "  Started $NUM_GROUPS schedulers (policy=$POLICY alpha=$ALPHA)"
}

start_provisioner() {
    local LOG_TAG="$1"
    POLL_PERIOD_S="$POLL_PERIOD" \
    GRPC_PORT=50300 \
    LATENCY_CSV="$LATENCY_CSV" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m provisioner.server > "$LOGDIR/${LOG_TAG}_provisioner.log" 2>&1 &
    echo "  Started provisioner (poll=${POLL_PERIOD}s)"
}

run_loadgen() {
    local TAG="$1"
    local OUTPUT="$RESULTS_DIR/${TAG}.csv"
    echo "  Running loadgen: duration=${DURATION}s rate=${ARRIVAL_RATE} -> $OUTPUT"
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m loadgen.client \
        --duration "$DURATION" \
        --arrival-distribution poisson \
        --arrival-rate "$ARRIVAL_RATE" \
        --trace-csv "$TRACE_CSV" \
        --output "$OUTPUT" 2>&1
}

echo "================================================================"
echo "  Quick Test: round_robin vs carbon_aware a=0.05"
echo "  Duration: ${DURATION}s  Arrival rate: ${ARRIVAL_RATE} req/s"
echo "  DCs/group: ${DCS_PER_GROUP}  Total DCs: $NUM_DCS  Schedulers: $NUM_GROUPS"
echo "  Nodes/DC: ${NUM_NODES}  TP: ${TP_SIZE}  Poll: ${POLL_PERIOD}s"
echo "================================================================"
echo ""

TESTS=(
    "round_robin|round_robin|0"
    "carbon_aware_a0.05|carbon_aware|0.05"
)

for TEST in "${TESTS[@]}"; do
    IFS='|' read -r TAG POLICY ALPHA <<< "$TEST"
    echo ""
    echo "--- Test: $TAG ---"

    kill_all
    start_dcs "$TAG"
    start_schedulers "$POLICY" "$ALPHA" "$TAG"
    start_provisioner "$TAG"

    sleep 8

    run_loadgen "$TAG"

    echo "  Collecting energy stats..."
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python scripts/sweep/collect_energy.py "$RESULTS_DIR/${TAG}_energy.json" \
        --requests-csv "$RESULTS_DIR/${TAG}.csv" 2>&1
    echo ""
done

echo ""
echo "================================================================"
echo "  All tests complete. Results in $RESULTS_DIR/"
echo "================================================================"
echo ""

# Print summary table
echo "Policy                      | OK  | Err | wall_avg | wall_p50 | wall_p99 | lat_avg  | lat_p50  | lat_p99  | Energy(kWh) | Carbon(g)"
echo "----------------------------|-----|-----|----------|----------|----------|----------|----------|----------|-------------|----------"

python3 << PYEOF
import csv, json, os

tests = [t.split('|') for t in """$(printf '%s\n' "${TESTS[@]}")""".strip().split('\n')]
results_dir = '$RESULTS_DIR'

for tag, policy, alpha in tests:
    req_csv = os.path.join(results_dir, f'{tag}.csv')
    energy_json = os.path.join(results_dir, f'{tag}_energy.json')
    if not os.path.isfile(req_csv):
        continue

    with open(req_csv) as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r['status'] == 'OK']
    err = [r for r in rows if r['status'] != 'OK']

    energy_kwh = None
    carbon_g = None
    try:
        with open(energy_json) as f:
            ej = json.load(f)
        energy_kwh = ej['total_energy_kwh']
        carbon_g = ej['total_carbon_g']
    except:
        pass

    if ok:
        def pct(vals, p):
            s = sorted(vals)
            return s[min(int(len(s) * p), len(s) - 1)]

        wall = sorted(float(r['wall_ms']) for r in ok)
        lat = sorted(float(r['latency_ms']) for r in ok)
        n = len(wall)
        e_str = f'{energy_kwh:.4f}' if energy_kwh is not None else 'N/A'
        c_str = f'{carbon_g:.1f}' if carbon_g is not None else 'N/A'
        pad = ' ' * (28 - len(tag))
        print(f'{tag}{pad}| {n:>3} | {len(err):>3} | {sum(wall)/n:>8.0f} | {pct(wall, 0.5):>8.0f} | {pct(wall, 0.99):>8.0f} | {sum(lat)/n:>8.0f} | {pct(lat, 0.5):>8.0f} | {pct(lat, 0.99):>8.0f} | {e_str:>11} | {c_str:>8}')
    else:
        pad = ' ' * (28 - len(tag))
        print(f'{tag}{pad}| {0:>3} | {len(err):>3} |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |         N/A |      N/A')
PYEOF

kill_all
echo "Done."

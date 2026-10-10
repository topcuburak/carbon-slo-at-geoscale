#!/usr/bin/env bash
# Sweep across scheduling policies and alpha values.
#
# 8 scheduler groups, configurable DCs per group (default 7).
# DC names match common/network.py SCHEDULER_GROUPS.
#
# Tests:
#   - round_robin, round_robin_local
#   - least_loaded, least_loaded_local
#   - weighted_least_loaded
#   - carbon_aware with alpha = 1.0, 0.9, 0.8, 0.5, 0.25
#
# Usage:
#   bash scripts/test/sweep_test.sh [duration] [arrival_rate] [dcs_per_group] [summary_csv]
#   bash scripts/test/sweep_test.sh 50 25                          # 50s at λ=25, 7 DCs/group
#   bash scripts/test/sweep_test.sh 120 40 8                       # 120s at λ=40, 8 DCs/group
#   bash scripts/test/sweep_test.sh 60 20 7 results/exp1.csv       # custom summary CSV path
#
# Carbon intensity:
#   SIM_DATETIME="2024-10-15T06" bash scripts/test/sweep_test.sh 50 25   # specific hour
#   SIM_DATETIME=random bash scripts/test/sweep_test.sh 50 25            # random hour (default)
#   SIM_YEAR=2023 bash scripts/test/sweep_test.sh 50 25                  # random hour from 2023

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/proto:$PYTHONPATH"

DURATION="${1:-50}"
ARRIVAL_RATE="${2:-25}"
DCS_PER_GROUP="${3:-7}"
LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv}"
POLL_PERIOD="${POLL_PERIOD_S:-2}"
LOGDIR="${LOGDIR:-results/logs}"
RESULTS_DIR="${RESULTS_DIR:-results/sweep}"
SUMMARY_CSV="${4:-results/summary.csv}"
CONCURRENT="${CONCURRENT:-5000}"
NUM_NODES="${NUM_NODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
SIM_DATETIME="${SIM_DATETIME:-2024-10-15T06}"
CARBON_DATA_ROOT="${CARBON_DATA_ROOT:-carbon_intensity}"

# Generated at runtime
ENDPOINTS="$LOGDIR/endpoints.json"
mkdir -p "$LOGDIR" "$RESULTS_DIR" "$(dirname "$SUMMARY_CSV")"

# ── Carbon intensities from real data ────────────────────────────────
# SIM_DATETIME controls the hour: "random" picks a random hour,
# "2024-10-15T06" uses that specific UTC hour.
declare -A CI
eval "$(CARBON_DATA_ROOT="$CARBON_DATA_ROOT" SIM_YEAR="${SIM_YEAR:-2024}" \
    python3 scripts/sweep/generate_ci.py "$SIM_DATETIME")"
echo "  CI snapshot: $SIM_DATETIME"

# ── DC lists per group (order matches common/network.py) ───────────
# First DCS_PER_GROUP from each list are used.
EUROPE_NORTH=(DK_East DK_West Estonia Finland GreatBritain Ireland Latvia Lithuania Netherlands Norway SE_North SE_Central SE_South Belgium Poland)
EUROPE_SOUTH=(Austria Czechia France Germany Hungary IT_North IT_CentralNorth IT_CentralSouth IT_South IT_Sardinia IT_Sicily Portugal Serbia Slovakia Slovenia Spain Switzerland Croatia)
US_EAST=(US_ISONE US_NYISO US_PJM US_MISO US_DukeCarolinas US_DukeProgressE US_DukeProgressW US_SCEG)
US_SOUTH=(US_Southern US_Tallahassee US_FLMunicipal US_FPL US_Gainesville US_TVA US_ERCOT US_SPP)
US_WEST=(US_APS US_Avista US_BANC US_BPA US_CAISO US_ElPaso US_IdahoPower US_LADWP US_NWEnergy US_PacifiCorpE US_PGE US_PSCo US_Turlock US_WAPA US_AssocElec)
MIDDLE_EAST=(Israel Turkey Cyprus Greece Bulgaria Romania SouthAfrica)
ASIA=(AU_NSW AU_NT AU_QLD AU_VIC IN_East IN_NorthEast IN_North IN_South IN_West JP_Chubu JP_Chugoku JP_Hokkaido JP_Hokuriku JP_Kansai JP_Okinawa JP_Tohoku JP_Tokyo PH_Luzon PH_Mindanao PH_Visayas Singapore SouthKorea Taiwan NewZealand)
SOUTH_AMERICA=(BR_Central BR_North BR_NorthEast BR_South Chile Peru Nicaragua)

ALL_GROUPS="europe_north europe_south us_east us_south us_west middle_east asia south_america"

# ── Select first N DCs per group and assign ports ──────────────────
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
    # Assign scheduler port
    SCHED_PORTS[$GROUP]=$SCHED_PORT
    SCHED_PORT=$((SCHED_PORT + 1))

    # Select DCs for this group
    GROUP_UPPER=$(echo "$GROUP" | tr '[:lower:]' '[:upper:]')
    SELECTED=$(select_dcs "$GROUP_UPPER" "$DCS_PER_GROUP")

    for DC in $SELECTED; do
        DC_PORTS[$DC]=$DC_PORT
        DC_PORT=$((DC_PORT + 1))
        ALL_DCS="$ALL_DCS $DC"
    done
done

ALL_DCS="${ALL_DCS# }"  # trim leading space
NUM_DCS=$(echo $ALL_DCS | wc -w)
NUM_GROUPS=$(echo $ALL_GROUPS | wc -w)

# ── Generate endpoints.json ────────────────────────────────────────
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

# ── Functions ────────────────────────────────────────────────────────

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
        --concurrent "$CONCURRENT" \
        --trace-csv "$LATENCY_CSV" \
        --output "$OUTPUT" 2>&1
}

# ── Main ─────────────────────────────────────────────────────────────
echo "================================================================"
echo "  gRPC Policy Sweep"
echo "  Duration: ${DURATION}s  Arrival rate: ${ARRIVAL_RATE} req/s"
echo "  DCs/group: ${DCS_PER_GROUP}  Total DCs: $NUM_DCS  Schedulers: $NUM_GROUPS"
echo "  Nodes/DC: ${NUM_NODES}  TP: ${TP_SIZE}  Poll: ${POLL_PERIOD}s"
echo "  CI snapshot: ${SIM_DATETIME} (year=${SIM_YEAR})"
echo "  Latency CSV: ${LATENCY_CSV}"
echo "  Summary CSV: ${SUMMARY_CSV}"
echo "  Endpoints: ${ENDPOINTS}"
echo "================================================================"
echo ""

# Define test cases: "tag|policy|alpha"
# Alpha is only used by carbon_aware; ignored by other policies.
TESTS=(
    "round_robin|round_robin|0"
    "round_robin_local|round_robin_local|0"
    "least_loaded|least_loaded|0"
    "least_loaded_local|least_loaded_local|0"
    "weighted_least_loaded|weighted_least_loaded|0"
    "carbon_aware_a1.0|carbon_aware|1.0"
    "carbon_aware_a0.9|carbon_aware|0.9"
    "carbon_aware_a0.8|carbon_aware|0.8"
    "carbon_aware_a0.7|carbon_aware|0.7"
    "carbon_aware_a0.6|carbon_aware|0.6"
    "carbon_aware_a0.5|carbon_aware|0.5"
    "carbon_aware_a0.4|carbon_aware|0.4"
    "carbon_aware_a0.3|carbon_aware|0.3"
    "carbon_aware_a0.2|carbon_aware|0.2"
    "carbon_aware_a0.15|carbon_aware|0.15"
    "carbon_aware_a0.1|carbon_aware|0.1"
    "carbon_aware_a0.05|carbon_aware|0.05"
    "carbon_aware_a0.0|carbon_aware|0.0"
)

for TEST in "${TESTS[@]}"; do
    IFS='|' read -r TAG POLICY ALPHA <<< "$TEST"
    echo ""
    echo "--- Test: $TAG ---"

    kill_all
    start_dcs "$TAG"
    start_schedulers "$POLICY" "$ALPHA" "$TAG"
    start_provisioner "$TAG"

    # Wait for services to initialize and provisioner to push first snapshot
    sleep 8

    run_loadgen "$TAG"

    # Collect energy/carbon stats from all DCs before killing them
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

# Print summary table + write summary CSV
echo "Policy                      | OK  | Err | wall_avg | wall_p50 | wall_p90 | wall_p99 | lat_avg  | lat_p50  | lat_p90  | lat_p99  | svc_avg  | svc_p50  | svc_p90  | svc_p99  | Energy(kWh) | Carbon(g)"
echo "----------------------------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------"

python3 << PYEOF
import csv, json, os

tests = [t.split('|') for t in """$(printf '%s\n' "${TESTS[@]}")""".strip().split('\n')]
results_dir = '$RESULTS_DIR'
summary_csv = '$SUMMARY_CSV'

summary_rows = []
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

    row = {
        'tag': tag, 'policy': policy, 'alpha': alpha,
        'ok': len(ok), 'errors': len(err),
    }

    if ok:
        def pct(vals, p):
            s = sorted(vals)
            return s[min(int(len(s) * p), len(s) - 1)]

        wall = sorted(float(r['wall_ms']) for r in ok)
        lat = sorted(float(r['latency_ms']) for r in ok)
        svc = sorted(float(r['service_ms']) for r in ok)
        n = len(wall)
        row.update({
            'wall_avg': round(sum(wall)/n, 1),
            'wall_p50': round(pct(wall, 0.5), 1), 'wall_p90': round(pct(wall, 0.9), 1), 'wall_p99': round(pct(wall, 0.99), 1),
            'lat_avg': round(sum(lat)/n, 1),
            'lat_p50': round(pct(lat, 0.5), 1), 'lat_p90': round(pct(lat, 0.9), 1), 'lat_p99': round(pct(lat, 0.99), 1),
            'svc_avg': round(sum(svc)/n, 1),
            'svc_p50': round(pct(svc, 0.5), 1), 'svc_p90': round(pct(svc, 0.9), 1), 'svc_p99': round(pct(svc, 0.99), 1),
        })
        e_str = f'{energy_kwh:.4f}' if energy_kwh is not None else 'N/A'
        c_str = f'{carbon_g:.1f}' if carbon_g is not None else 'N/A'
        pad = ' ' * (28 - len(tag))
        print(f'{tag}{pad}| {n:>3} | {len(err):>3} | {row["wall_avg"]:>8.0f} | {row["wall_p50"]:>8.0f} | {row["wall_p90"]:>8.0f} | {row["wall_p99"]:>8.0f} | {row["lat_avg"]:>8.0f} | {row["lat_p50"]:>8.0f} | {row["lat_p90"]:>8.0f} | {row["lat_p99"]:>8.0f} | {row["svc_avg"]:>8.0f} | {row["svc_p50"]:>8.0f} | {row["svc_p90"]:>8.0f} | {row["svc_p99"]:>8.0f} | {e_str:>11} | {c_str:>8}')
    else:
        pad = ' ' * (28 - len(tag))
        print(f'{tag}{pad}| {0:>3} | {len(err):>3} |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |      N/A |         N/A |      N/A')

    row['energy_kwh'] = round(energy_kwh, 4) if energy_kwh is not None else ''
    row['carbon_g'] = round(carbon_g, 1) if carbon_g is not None else ''
    summary_rows.append(row)

# Write summary CSV
if summary_rows:
    fieldnames = [
        'tag', 'policy', 'alpha', 'ok', 'errors',
        'wall_avg', 'wall_p50', 'wall_p90', 'wall_p99',
        'lat_avg', 'lat_p50', 'lat_p90', 'lat_p99',
        'svc_avg', 'svc_p50', 'svc_p90', 'svc_p99',
        'energy_kwh', 'carbon_g',
    ]
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nSummary CSV written to: {summary_csv}')
PYEOF

kill_all
echo "Done."

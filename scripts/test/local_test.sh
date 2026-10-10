#!/usr/bin/env bash
# Launch all gRPC services locally for a quick manual test.
# Usage: bash scripts/test/local_test.sh [alpha] [dcs_per_group]
#   alpha: routing alpha (default 0.5)
#   dcs_per_group: DCs per scheduler group (default 3)
#
# Starts: N×8 DCs + 8 schedulers + 1 provisioner
# Then run loadgen manually:
#   STATIC_ENDPOINTS_FILE=/tmp/grpc_test_logs/endpoints.json \
#     python -m loadgen.client --duration 30 --arrival-distribution poisson --arrival-rate 2

set -e
cd "$(dirname "$0")/../.."

ALPHA="${1:-0.5}"
DCS_PER_GROUP="${2:-3}"
LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv}"
POLL_PERIOD="${POLL_PERIOD_S:-2}"
NUM_NODES="${NUM_NODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
LOGDIR="/tmp/grpc_test_logs"
ENDPOINTS="$LOGDIR/endpoints.json"

mkdir -p "$LOGDIR"

# Kill any previous run
pkill -f "python -m datacenter.server" 2>/dev/null || true
pkill -f "python -m scheduler.server" 2>/dev/null || true
pkill -f "python -m provisioner.server" 2>/dev/null || true
sleep 1

# ── Carbon intensities from real data ────────────────────────────────
SIM_DATETIME="${SIM_DATETIME:-2024-10-15T06}"
CARBON_DATA_ROOT="${CARBON_DATA_ROOT:-carbon_intensity}"
declare -A CI
eval "$(CARBON_DATA_ROOT="$CARBON_DATA_ROOT" SIM_YEAR="${SIM_YEAR:-2024}" \
    python3 scripts/sweep/generate_ci.py "$SIM_DATETIME")"
echo "  CI snapshot: $SIM_DATETIME"

# ── DC lists per group (order matches common/network.py) ──────────
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

# ── Generate endpoints.json ────────────────────────────────────────
{
    echo "{"
    first=true
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
} > "$ENDPOINTS"

# ── Start DCs ──────────────────────────────────────────────────────
echo "=== Starting $NUM_DCS datacenters ($DCS_PER_GROUP per group) ==="

for DC in $ALL_DCS; do
    DC_ID="$DC" \
    GRPC_PORT="${DC_PORTS[$DC]}" \
    CARBON_INTENSITY="${CI[$DC]}" \
    LATENCY_CSV="$LATENCY_CSV" \
    NUM_NODES="$NUM_NODES" \
    TP_SIZE="$TP_SIZE" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m datacenter.server > "$LOGDIR/dc_${DC}.log" 2>&1 &
    echo "  DC $DC on port ${DC_PORTS[$DC]} (CI=${CI[$DC]})"
done

# ── Start schedulers ───────────────────────────────────────────────
echo "=== Starting 8 schedulers ==="

for GROUP in $ALL_GROUPS; do
    SCHED_ID="sched-${GROUP}" \
    SCHEDULER_GROUP="$GROUP" \
    ROUTING_POLICY="carbon_aware" \
    ROUTING_ALPHA="$ALPHA" \
    SIMULATE_RTT="true" \
    GRPC_PORT="${SCHED_PORTS[$GROUP]}" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
      python -m scheduler.server > "$LOGDIR/sched_${GROUP}.log" 2>&1 &
    echo "  Scheduler sched-${GROUP} on port ${SCHED_PORTS[$GROUP]}"
done

# ── Start provisioner ──────────────────────────────────────────────
echo "=== Starting provisioner ==="

POLL_PERIOD_S="$POLL_PERIOD" \
GRPC_PORT=50300 \
LATENCY_CSV="$LATENCY_CSV" \
STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
  python -m provisioner.server > "$LOGDIR/provisioner.log" 2>&1 &
echo "  Provisioner on port 50300 (poll=${POLL_PERIOD}s)"

sleep 3
echo ""
echo "=== All services started. Logs in $LOGDIR ==="
echo ""
echo "Run loadgen:"
echo "  STATIC_ENDPOINTS_FILE=$ENDPOINTS \\"
echo "    python -m loadgen.client --duration 30 --arrival-distribution poisson --arrival-rate 2 --concurrent 64"
echo ""
echo "To stop all services:"
echo "  pkill -f 'python -m datacenter.server'; pkill -f 'python -m scheduler.server'; pkill -f 'python -m provisioner.server'"

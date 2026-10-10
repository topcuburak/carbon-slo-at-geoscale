#!/usr/bin/env bash
#
# Start the provisioner: poll every datacentre, push a snapshot to every
# scheduler, on a fixed cadence.
#
#   provisioner/start.sh
#   provisioner/start.sh --port 50060
#   provisioner/start.sh --mode async --waves 4 --wave-interval 0.5
#
# Start it AFTER the datacentres and schedulers: it discovers both from the
# endpoint map the two launchers merge into, and a peer missing from that map
# cannot be polled or pushed to even once it is running.
#
# There is no config file. Every setting is an environment variable the server
# already reads, which is how the sweep drives it too; the flags below are a
# readable front end for the same variables.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${GRPC_PORT:-50053}"
ENDPOINTS="${STATIC_ENDPOINTS_FILE:-results/logs/endpoints.json}"
MODE="${POLL_MODE:-sync}"
PERIOD="${POLL_PERIOD_S:-1}"
WAVES="${POLL_WAVE_COUNT:-4}"
WAVE_INTERVAL="${POLL_WAVE_INTERVAL_S:-0.5}"
BETA="${RTT_BETA:-0.5}"
# How heat is normalised: "cap" against a fixed per-engine batch (load shows up
# in RTT), "peak" against the busiest group (the pre-2026-09-02 behaviour, blind
# to the level). RTT_HEAT_CAP should track the engines' max_num_seqs.
NORM="${RTT_NORM:-cap}"
HEAT_CAP="${RTT_HEAT_CAP:-32}"
DISCOVERY="${DISCOVERY_REFRESH_S:-10}"
VERB="${VERBOSITY:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)          PORT="$2"; shift 2 ;;
        --endpoints)     ENDPOINTS="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --period)        PERIOD="$2"; shift 2 ;;
        --waves)         WAVES="$2"; shift 2 ;;
        --wave-interval) WAVE_INTERVAL="$2"; shift 2 ;;
        --rtt-beta)      BETA="$2"; shift 2 ;;
        --rtt-norm)      NORM="$2"; shift 2 ;;
        --rtt-heat-cap)  HEAT_CAP="$2"; shift 2 ;;
        --discovery)     DISCOVERY="$2"; shift 2 ;;
        --verbosity)     VERB="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

# sync and async are not a tuning knob: they are the comparison the polling
# design exists for. sync pushes one identical snapshot to every scheduler, so
# they all see the same state and herd; async staggers the fleet across waves
# so they do not. The server refuses anything else rather than falling back.
case "$MODE" in
    sync|async) ;;
    *) echo "--mode must be sync or async, got '$MODE'" >&2; exit 1 ;;
esac
case "$NORM" in
    cap|peak) ;;
    *) echo "--rtt-norm must be cap or peak, got '$NORM'" >&2; exit 1 ;;
esac

[[ -f "$ENDPOINTS" ]] || {
    echo "no endpoint map at $ENDPOINTS -- start the datacentres and" >&2
    echo "schedulers first, or pass --endpoints" >&2
    exit 1
}

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/proto:${PYTHONPATH:-}"

n_dc=$(python3 -c "
import json; d = json.load(open('$ENDPOINTS'))
print(sum(1 for k in d if not k.startswith('sched-') and k != 'provisioner'))")
n_sched=$(python3 -c "
import json; d = json.load(open('$ENDPOINTS'))
print(sum(1 for k in d if k.startswith('sched-')))")

if [[ "$MODE" == "async" ]]; then
    cycle=$(python3 -c "print($WAVES * $WAVE_INTERVAL)")
    echo "=== provisioner  port=$PORT  mode=async  waves=$WAVES x ${WAVE_INTERVAL}s (cycle ${cycle}s)  ==="
else
    echo "=== provisioner  port=$PORT  mode=sync  poll=${PERIOD}s  ==="
fi
echo "    endpoints=$ENDPOINTS  ($n_dc datacentres, $n_sched schedulers)"
echo "    rtt: beta=$BETA  norm=$NORM  heat_cap=$HEAT_CAP"

exec env \
    GRPC_PORT="$PORT" \
    STATIC_ENDPOINTS_FILE="$ENDPOINTS" \
    POLL_PERIOD_S="$PERIOD" \
    POLL_MODE="$MODE" \
    POLL_WAVE_COUNT="$WAVES" \
    POLL_WAVE_INTERVAL_S="$WAVE_INTERVAL" \
    RTT_BETA="$BETA" \
    RTT_NORM="$NORM" \
    RTT_HEAT_CAP="$HEAT_CAP" \
    DISCOVERY_REFRESH_S="$DISCOVERY" \
    VERBOSITY="$VERB" \
    python3 -m provisioner.server

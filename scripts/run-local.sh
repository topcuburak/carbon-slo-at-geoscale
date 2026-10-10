#!/usr/bin/env bash
#
# Launch the full distributed AI inference system locally using tmux.
# Topology is read from env-local.sh. Each component runs in its own
# tmux window via the individual start-* scripts.
#
# Usage:
#   ./scripts/run-local.sh              # launch all components
#   ./scripts/run-local.sh --kill       # tear down the tmux session
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env-local.sh"

SESSION="dist-ai"

# ---------------------------------------------------------------------------
# Kill mode
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--kill" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Session '$SESSION' killed." || echo "No session to kill."
    exit 0
fi

echo "=== Distributed AI Inference — Local Launch ==="
echo "  Datacenters:  $NUM_DCS  (ports $DC_BASE_PORT-$((DC_BASE_PORT + NUM_DCS - 1)))"
echo "  Schedulers:   $NUM_SCHEDULERS  (ports $SCHED_BASE_PORT-$((SCHED_BASE_PORT + NUM_SCHEDULERS - 1)))"
echo "  Provisioner:  port $PROV_PORT"
echo "  tmux session: $SESSION"
echo ""

# ---------------------------------------------------------------------------
# Create tmux session
# ---------------------------------------------------------------------------
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "overview"
tmux send-keys -t "$SESSION:overview" \
    "echo '=== dist-ai session ==='; echo 'Ctrl-b w = list windows | Ctrl-b n/p = next/prev | Ctrl-b & = kill window'" Enter

# ---------------------------------------------------------------------------
# Datacenters
# ---------------------------------------------------------------------------
# One window for the whole fleet, not one per site: the launcher starts every
# datacentre in the topology, waits for all of them to bind, and refuses to
# hand back a fleet with a hole in it. The per-index loop this replaced called
# a script that ignored its argument and started them all anyway -- and could
# not start any of them, because it sourced env-local.sh from its own directory
# rather than scripts/.
tmux new-window -t "$SESSION" -n "datacenters"
tmux send-keys -t "$SESSION:datacenters" \
    "$ROOT_DIR/datacenter/start.sh --foreground" Enter

sleep 5  # give the fleet time to bind its ports

# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
# One window for the fleet, and no index: scheduler/start.sh reads
# scheduler/fleet.yaml and starts one process per topology group, waiting for
# all of them. The loop this replaces passed $i to a script that has no such
# argument -- it would have been forwarded to launch.py and rejected -- and is
# the same mistake the datacentre loop above carried.
tmux new-window -t "$SESSION" -n "schedulers"
tmux send-keys -t "$SESSION:schedulers" "$ROOT_DIR/scheduler/start.sh" Enter

sleep 1

# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n "provisioner"
tmux send-keys -t "$SESSION:provisioner" "$ROOT_DIR/provisioner/start.sh" Enter

sleep 1

# ---------------------------------------------------------------------------
# LoadGen (ready to run, waiting for user)
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n "loadgen"
tmux send-keys -t "$SESSION:loadgen" \
    "echo 'Run the load generator with:'; echo '  $ROOT_DIR/loadgen/start.sh --duration 30'" Enter

# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------
tmux select-window -t "$SESSION:overview"
echo ""
echo "All components launched. Attaching to tmux session..."
echo "  Ctrl-b w    — list all windows"
echo "  Ctrl-b n/p  — next/previous window"
echo "  Ctrl-b &    — kill current window"
echo ""
tmux attach -t "$SESSION"

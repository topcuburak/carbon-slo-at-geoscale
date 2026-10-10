#!/usr/bin/env bash
#
# Start the scheduler fleet from a YAML spec.
#
#   scheduler/start.sh
#   scheduler/start.sh --policy carbon_aware --alpha 0.5
#   scheduler/start.sh --config runs/my_fleet.yaml --groups europe_north,us_east
#
# The YAML becomes flags; anything on the command line is appended after and
# therefore wins. Unknown keys are refused by name.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="scheduler/fleet.yaml"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

# Command substitution inside `if !`, not process substitution: the latter
# discards the child's exit status, so a rejected config would sail past
# `set -e` and the fleet would start with no flags at all.
if ! CFG_TEXT="$(python3 - "$CONFIG" <<'PY'
import sys, yaml
KEYS = {"groups", "base_port", "policy", "alpha", "ilp_max_iterations",
        "osl_source", "ref_osl",
        "model_name", "tp_size", "max_num_seqs",
        "batch_window_ms",
        "ttft_budget_ms", "tpot_budget_ms", "simulate_rtt", "verbosity",
        "endpoints_file", "log_dir", "log_tag"}
cfg = (yaml.safe_load(open(sys.argv[1])) or {}).get("fleet") or {}
unknown = set(cfg) - KEYS
if unknown:
    sys.exit(f"unknown key(s) in {sys.argv[1]}: {', '.join(sorted(unknown))}. "
             f"Known: {', '.join(sorted(KEYS))}")
for k, v in cfg.items():
    if v is None or v == "":
        continue
    print(f"--{k.replace('_', '-')}")
    print(v)
PY
)"; then
    exit 1
fi
CFG_ARGS=()
[[ -n "$CFG_TEXT" ]] && mapfile -t CFG_ARGS <<< "$CFG_TEXT"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/proto:${PYTHONPATH:-}"

echo "=== schedulers  config=$CONFIG ==="
exec python3 -m scheduler.launch "${CFG_ARGS[@]}" "${ARGS[@]+"${ARGS[@]}"}"

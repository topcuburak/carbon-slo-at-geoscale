#!/usr/bin/env bash
#
# Start the datacentre fleet from a YAML spec.
#
#   datacenter/start.sh
#   datacenter/start.sh --model-name qwen2.5_7B --tp-size 2
#   datacenter/start.sh --config runs/my_fleet.yaml --groups europe_north
#   datacenter/start.sh --foreground        # Ctrl-C stops the fleet
#
# The YAML becomes flags; anything on the command line is appended after and
# therefore wins. Unknown keys are refused by name.
#
# Companion to start_schedulers.sh. Start the datacentres FIRST: a scheduler
# reads the endpoint map at startup to learn which sites it owns, and both
# launchers merge into the same file rather than overwriting each other.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="datacenter/fleet.yaml"
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
KEYS = {"groups", "base_port", "num_nodes", "num_gpus_per_node", "tp_size",
        "max_batch_size", "max_model_len", "model_name", "latency_mode",
        "latency_csv", "sim_datetime", "default_ci", "endpoints_file",
        "log_dir", "log_tag", "bs_profile_dir", "verbosity", "wait_s"}
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

echo "=== datacentres  config=$CONFIG ==="
exec python3 -m datacenter.launch "${CFG_ARGS[@]}" "${ARGS[@]+"${ARGS[@]}"}"

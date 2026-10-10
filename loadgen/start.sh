#!/usr/bin/env bash
#
# Run the load generator standalone against a locally running fleet.
#
# Configuration comes from a YAML file; anything on the command line is appended
# after it and therefore wins. The resolved command is printed and saved beside
# the results, so what ran is always recoverable.
#
#   loadgen/start.sh                                # loadgen/config.yaml
#   loadgen/start.sh --arrival-rate 50 --duration 60
#   loadgen/start.sh --config runs/long_qwen.yaml
#   loadgen/start.sh --trace-csv loadgen/scenarios/long_qwen2.5_7B.csv
#
# The script supplies the three things the client cannot work out for itself:
# where the schedulers are, an import path that includes the generated
# protobufs, and a worker pool wide enough to stay open-loop.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="loadgen/config.yaml"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

# YAML -> flags. Every key must map to a real client flag: a key that does not
# is a typo or a leftover, and silently ignoring it is how the previous
# config.yaml came to advertise settings nothing read.
# Command substitution, not process substitution: `mapfile < <(...)` throws
# the child's exit status away, so a rejected config would sail past `set -e`
# and the client would start with no flags at all.
if ! CFG_TEXT="$(python3 - "$CONFIG" <<'PY'
import sys, yaml
KEYS = {"duration", "arrival_rate", "arrival_distribution", "seed",
        "min_interval_ms", "max_interval_ms", "trace_csv", "concurrent",
        "output", "verbosity", "tag"}
cfg = (yaml.safe_load(open(sys.argv[1])) or {}).get("loadgen") or {}
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

# Endpoints: an explicit file or JSON wins; otherwise build the map a local run
# uses. scripts/env-local.sh is where that lives -- this script used to source
# it from its own directory, where there is no such file, so `set -e` killed it
# on line one.
if [[ -z "${STATIC_ENDPOINTS_FILE:-}" && -z "${STATIC_ENDPOINTS_JSON:-}" ]]; then
    # The map the other three launchers merge into. Prefer it: env-local.sh
    # describes a two-datacentre local topology, and the real one is 104 across
    # 8 groups, so falling through to it silently addresses a fleet that is not
    # the one running.
    DEFAULT_EP="$PROJECT_DIR/results/logs/endpoints.json"
    if [[ -f "$DEFAULT_EP" ]]; then
        export STATIC_ENDPOINTS_FILE="$DEFAULT_EP"
        n=$(python3 -c "
import json; d=json.load(open('$DEFAULT_EP'))
print(sum(1 for k in d if k.startswith('sched-')))")
        echo "    endpoints=$DEFAULT_EP  ($n schedulers)"
    else
        echo "    no $DEFAULT_EP -- start the schedulers first, or pass" >&2
        echo "    STATIC_ENDPOINTS_FILE. Falling back to scripts/env-local.sh," >&2
        echo "    which describes a 2-datacentre local topology." >&2
        source "$PROJECT_DIR/scripts/env-local.sh"
        export STATIC_ENDPOINTS_JSON="$SCHED_ENDPOINTS"
    fi
fi

# The client imports `loadgen.*` and the generated `inference_pb2`, so both the
# repo root and proto/ have to be importable. `python3 loadgen/client.py` puts
# only loadgen/ on the path and dies at the first import; it has to be `-m`.
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/proto:${PYTHONPATH:-}"

echo "=== loadgen  config=$CONFIG ==="
printf '    %s' "python3 -m loadgen.client"
printf ' %q' "${CFG_ARGS[@]}" "${ARGS[@]+"${ARGS[@]}"}"
printf '\n'

# Record what actually ran next to the results, since the command line and the
# config file can disagree and the flags are what decided it.
OUT_DIR="results/sweeps/per_request"
mkdir -p "$OUT_DIR"
{ printf '%s' "python3 -m loadgen.client"
  printf ' %q' "${CFG_ARGS[@]}" "${ARGS[@]+"${ARGS[@]}"}"
  printf '\n'; } > "$OUT_DIR/last_command.txt"

exec python3 -m loadgen.client "${CFG_ARGS[@]}" "${ARGS[@]+"${ARGS[@]}"}"

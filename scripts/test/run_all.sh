#!/usr/bin/env bash
#
# Every assertion suite, in one run. No fleet, no ports, no GPUs -- each suite
# stubs whatever it talks to, so a failure here is the simulator's logic and
# not the environment.
#
#   scripts/test/run_all.sh
#
# The .sh files beside these are a different kind of test: they start real
# processes and drive a real run. Those need a machine that can bind ports and
# take minutes; these take about one.
#
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/proto:${PYTHONPATH:-}"

# Regenerate the proto stubs when they are older than the schema.
#
# The generated _pb2 files are gitignored, so a `git pull` that changes
# inference.proto leaves the OLD ones sitting there and every suite below then
# tests against message definitions that no longer exist. Nothing announces it:
# the failures surface as "InferenceRequest has no req_id" scattered across
# four suites, which reads like a code fault and is not one. This cost a real
# detour on a host that was 103 commits behind, so the check is here rather
# than in the instructions.
_PROTO=proto/inference.proto
_STUB=proto/inference_pb2.py
if [[ ! -f "$_STUB" || "$_PROTO" -nt "$_STUB" ]]; then
    if python3 -c "import grpc_tools.protoc" 2>/dev/null; then
        echo "── regenerating proto stubs (schema is newer)"
        python3 -m grpc_tools.protoc -I proto \
            --python_out=proto --grpc_python_out=proto "$_PROTO" || {
            echo "   protoc failed; the suites below would test stale stubs" >&2
            exit 1
        }
    else
        echo "   WARNING: $_STUB is older than $_PROTO and grpc_tools is not" >&2
        echo "   installed, so every suite below tests an out-of-date schema." >&2
        echo "   pip install grpcio-tools" >&2
    fi
fi

SUITES=(
    scripts/test/test_inference_engine.py   # one engine: batching, energy, tokens
    scripts/test/test_datacenter.py         # nodes, placement, energy, the RPCs
    scripts/test/test_scheduler.py          # policies, snapshots, OSL, batching
    scripts/test/test_forwarding.py         # the cross-region hop
    scripts/test/test_subgroups.py          # several schedulers in one region
    scripts/test/test_provisioner.py        # polling, congestion, snapshots
    scripts/test/test_loadgen.py            # arrivals, trace, the request row
    scripts/test/test_ilp_policies.py       # what the solver actually decides
    scripts/test/test_perf_models.py        # the latency and energy fits
    scripts/test/test_carbon.py             # the archive every result multiplies by
    scripts/test/test_config.py             # what a run actually ran with
)

failed=()
total_pass=0
total_all=0

for s in "${SUITES[@]}"; do
    echo "── $(basename "$s")"
    out="$(python3 "$s" 2>&1)"
    rc=$?
    # Each suite ends with "N/M passed".
    line="$(printf '%s\n' "$out" | grep -oE '[0-9]+/[0-9]+ passed' | tail -1)"
    if [[ -n "$line" ]]; then
        total_pass=$(( total_pass + ${line%%/*} ))
        rest="${line#*/}"
        total_all=$(( total_all + ${rest%% *} ))
    fi
    if [[ $rc -ne 0 ]]; then
        failed+=("$(basename "$s")")
        # Only the failures, not the whole log.
        printf '%s\n' "$out" | grep -E "^  (FAIL|ERROR)" -A 1
    fi
    echo "   ${line:-no result line -- the suite did not finish}"
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "$total_pass/$total_all assertions passed across ${#SUITES[@]} suites"
else
    echo "$total_pass/$total_all assertions passed; FAILED: ${failed[*]}"
    exit 1
fi

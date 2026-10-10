#!/usr/bin/env bash
# =============================================================================
#  Sweep all scheduling policies across arrival rates
# =============================================================================
#
# Runs the full experiment: 4 schedulers, 20 DCs, LLaMA-8B TP=1
# with mixed heterogeneous trace across all scheduling policies.
#
# Usage:
#   bash scripts/sweep/sweep_all_policies.sh [--seed 42] [--duration 120]
#
# Environment overrides:
#   SEED=42 DURATION=120 TTFT_BUDGET=1500 TPOT_BUDGET=60 bash scripts/sweep/sweep_all_policies.sh
#
# TTFT_BUDGET is the additive *overhead slack* in ms; the per-request
# TTFT budget is computed as
#     budget(ISL) = TTFT_BUDGET + 1.5 · analytical_prefill_ms(ISL)
# inside the scheduler and the analyzer. TPOT_BUDGET is a flat ms cap.
# =============================================================================

set -euo pipefail

# ── Interpreter preflight ──
# Every process this script launches is `python ...`, and every one of them is
# launched into the background with its output going to a per-role log. So a
# missing interpreter does not fail here -- it fails silently in thirteen
# children, and the first symptom is the datacentre readiness probe giving up:
#
#     ABORT: DC AU_VIC never opened port 50100 (waited 120s)
#
# which names the wrong layer and costs two minutes to say nothing. Measured
# 2026-09-07, from a shell where the conda env had not been activated.
if ! command -v python >/dev/null 2>&1; then
    echo "ABORT: no 'python' on PATH." >&2
    echo "  Every simulator process is launched as 'python -m ...', so this" >&2
    echo "  would fail in each of them separately and surface as a" >&2
    echo "  datacentre that never opened its port." >&2
    echo "  On our testbed: conda activate osl" >&2
    exit 1
fi
if ! python -c "import grpc" >/dev/null 2>&1; then
    echo "ABORT: python cannot import 'grpc'." >&2
    echo "  The interpreter on PATH is not the one the simulator needs" >&2
    echo "  ($(command -v python)). On our testbed: conda activate osl" >&2
    exit 1
fi
# inference_pb2 is NOT checked here: the stubs are gitignored and the block
# below is what generates them, so a fresh checkout legitimately lacks them.
# It is checked after that block instead.

# ── Proto stub freshness check ──
# The generated proto/inference_pb2*.py files are gitignored, so stale
# stubs are a common source of cryptic runtime errors on remote machines
# ("Protocol message has no field X"). Regenerate if the .proto source
# is newer than the generated Python, or the generated file is missing.
_PROTO_SRC="proto/inference.proto"
_PROTO_PY="proto/inference_pb2.py"
_PROTO_GRPC_PY="proto/inference_pb2_grpc.py"
if [ ! -f "$_PROTO_PY" ] || [ ! -f "$_PROTO_GRPC_PY" ] || \
   [ "$_PROTO_SRC" -nt "$_PROTO_PY" ] || \
   [ "$_PROTO_SRC" -nt "$_PROTO_GRPC_PY" ]; then
    echo "→ Regenerating proto stubs ($_PROTO_SRC is newer or stubs missing)..."
    if ! python -c "import grpc_tools" 2>/dev/null; then
        echo "  grpc_tools is missing. The generated stubs are NOT in git, so a" >&2
        echo "  machine that pulls a changed .proto has to regenerate them. Install" >&2
        echo "  the version that matches this environment's grpcio, or the stubs" >&2
        echo "  and the runtime disagree:" >&2
        echo "" >&2
        echo "    pip install \"grpcio-tools==\$(python -c 'import grpc;print(grpc.__version__)')\"" >&2
        echo "" >&2
        exit 1
    fi
    python -m grpc_tools.protoc -I proto \
        --python_out=proto --grpc_python_out=proto "$_PROTO_SRC"
    echo "  done."
fi

# Now the stubs must exist and be importable -- checked here rather than
# before the block above, which is what creates them on a fresh checkout.
if ! PYTHONPATH=proto:. python -c "import inference_pb2" >/dev/null 2>&1; then
    echo "ABORT: python cannot import 'inference_pb2' even after the stub" >&2
    echo "  regeneration above. Check proto/ and the interpreter" >&2
    echo "  ($(command -v python))." >&2
    exit 1
fi

# ── Configuration ──
SEED="${SEED:-42}"
DURATION="${DURATION:-120}"
TTFT_BUDGET="${TTFT_BUDGET:-1500}"
TPOT_BUDGET="${TPOT_BUDGET:-60}"
# Whose TPOT the feasibility test is about: the arriving request
# ("arrival", every result so far) or the batch already running at the
# datacentre ("batch"). See scheduler/policies/common/latency_estimates.py.
TPOT_FILTER="${TPOT_FILTER:-arrival}"
# Hard cap, PER ENGINE: a site is full when active + pending reaches
# bs_cap x its engine count (capacity.dc_capacity). 0 disables.
#
# The warning that used to stand here -- "setting it to max_num_seqs throttles
# a site to 1/engines of its real capacity", from a run where BS_CAP=128 left
# every site permanently full and the ILP seated nothing -- described the older
# semantics, where the number was a site-wide total. dc_capacity multiplies by
# dc_engines now, so BS_CAP=max_num_seqs is exactly the engines' own admission
# limit rather than 1/engines of it.
#
# It is not redundant with the SLA filter, which is the other brake. That
# filter rejects a site only when the TTFT ESTIMATE exceeds the budget, and
# under concentration the estimate is where it is weakest: measured at
# 4400 req/s, carbon_aware:0.0 (alpha is the LATENCY weight, so 0.0 is
# carbon-only and has no load term at all) lost 53.2% of its requests, and
# least_loaded 6.6% with 2 of 16 sites at bs 177 against a 192 cap and 1113
# queued while the other 14 sat at zero. Neither was the scheduler's ceiling.
# A cap on what the engines can physically admit does not depend on a model
# being right.
BS_CAP="${BS_CAP:-0}"
# Provisioner poll mode: "sync" (current, herd-prone) or "async"
# (staggered waves, breaks herd symmetry). In async mode the full
# refresh cycle takes POLL_WAVE_COUNT × POLL_WAVE_INTERVAL_S seconds
# with schedulers receiving partial snapshots every wave.
POLL_MODE="${POLL_MODE:-sync}"
POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}"
POLL_WAVE_INTERVAL_S="${POLL_WAVE_INTERVAL_S:-0.5}"
# The RTT congestion model normalises per-DC load against this. It used to
# be fed MAX_BATCH_SIZE, which tied the network model to the engines' seat
# count: changing the seats silently changed the RTT matrix, so a capacity
# experiment moved two things at once. They are unrelated quantities.
RTT_HEAT_CAP="${RTT_HEAT_CAP:-32}"
MODEL_NAME="${MODEL_NAME:-llama3_8B}"
TP_SIZE="${TP_SIZE:-2}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
# The engine's max_model_len. It sets the prefill token budget
# (max_num_batched_tokens = max(max_model_len, 2048), vLLM's rule with chunked
# prefill off), which decides how many queued prompts fuse into one forward
# pass and how many passes a long prompt is split across. The trace here is 45%
# above 2048 with a p90 of 12190 tokens, so the testbed's 2048 would split one
# pass into six and charge the fixed cost of a prefill step six times. 32768 is
# also where the latency and energy calibration envelopes stop -- do not raise
# it without extending them.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Where the scheduler gets a request's output length when it prices a
# placement: "oracle" is the trace's true out_tok, which makes the scheduler
# clairvoyant and is the configuration every previous result was produced
# under; "predicted" uses the trace's predicted_osl column, which only some
# traces carry; "ref" is a flat REF_OSL for every request. Run the sweep both
# ways on a trace that carries both columns -- every mixed_/mlperf_/long_ trace does
# does -- to report what the oracle is worth.
OSL_SOURCE="${OSL_SOURCE:-oracle}"
REF_OSL="${REF_OSL:-500}"
BATCH_WINDOW_MS="${BATCH_WINDOW_MS:-100}"
# Requests that wake an ILP batch before its window is up. A trigger, not a
# drain cap; see scheduler/config_loader.py for why 300 and not 500.
BATCH_WINDOW_SIZE="${BATCH_WINDOW_SIZE:-300}"
ILP_MAX_ITER="${ILP_MAX_ITER:-3}"
VERBOSITY="${VERBOSITY:-1}"
DC_VERBOSITY="${DC_VERBOSITY:-0}"

# GPU latency mode for datacenter servers. The scheduler always uses
# "analytical" for predictions. The DC can use either:
#   analytical → fitted formulas (same model as scheduler)
#   lookup     → interpolated from benchmark CSVs (closer to real hardware)
DC_LATENCY_MODE="${DC_LATENCY_MODE:-analytical}"
SIMULATE_RTT="${SIMULATE_RTT:-true}"
# Default is the mixed family for the model being swept -- every source the
# paper draws on, long prompts included, and it carries predicted_osl so
# OSL_SOURCE=predicted works without swapping traces. The other two families
# are mlperf_<model>.csv (what MLPerf runs for that model) and long_<model>.csv
# (GovReport only, ISL floored at 2048); see loadgen/scenarios/README.md.
# Was mixed_trace_3ds.csv, which predates the nine per-model traces and has no
# predicted_osl column.
TRACE_CSV="${TRACE_CSV:-loadgen/scenarios/mixed_${MODEL_NAME}.csv}"

# Latency data — env-overridable so non-LLaMA models can plug in their CSVs.
# DC-side TTFT predictions come from the model-specific JSON lookups loaded
# via MODEL_NAME, so LATENCY_CSV is mainly a placeholder satisfying the
# LatencyLookup constructor and providing fallbacks for BS=1 ttft_by_isl.
# preferred (Mixtral has its own; DeepSeek uses Mixtral's as placeholder).
LATENCY_CSV="${LATENCY_CSV:-datacenter/perf_models/lookup/measured/full/llama3.1-8B/all_datasets_merged_benchmark.csv}"

# Arrival rates to sweep
# Override with: ARRIVAL_RATES="10 15 20 25" bash scripts/sweep/sweep_all_policies.sh
if [ -n "${ARRIVAL_RATES:-}" ]; then
    read -ra ARRIVAL_RATES <<< "$ARRIVAL_RATES"
else
    ARRIVAL_RATES=(30 35 40 50)
fi

# Carbon-intensity snapshot timestamp (ISO 8601, UTC). Used by the inline
# Python that calls CarbonLoader.get_ci(). Override to study temporal
# variation: SIM_DATETIME="2024-07-15T18:00:00"
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"

# Policies to test
# Baselines (no SLA):
#   round_robin, round_robin_local, least_loaded, least_loaded_local, weighted_least_loaded
# Carbon-aware (alpha sweep, with SLA):
#   carbon_aware -- and read the direction before choosing values.
#
#   ALPHA IS THE LATENCY WEIGHT, NOT THE CARBON WEIGHT:
#       score = alpha * norm_latency + (1 - alpha) * norm_carbon,  minimised
#   so alpha=1.0 ignores carbon entirely and alpha=0.0 ignores latency. The
#   default below used to be 0.5/0.75/0.9/0.95, which is the latency-dominant
#   HALF of the range and reads like a carbon sweep. Measured 2026-09-02: every
#   value in it served 49-51% of requests on the clean sites -- i.e. the same
#   as round_robin -- while alpha=0.0 holds a green site at any load. Sweeping
#   only 0.5+ hides the frontier the policy actually has.
# SLA-aware carbon-optimal:
#   ilp_carbon_sla, iterative_ilp_carbon_sla, facility_ilp_carbon_sla

# Override with: POLICIES="iterative_ilp_carbon_sla" bash scripts/sweep/sweep_all_policies.sh
# Space-separated list. Supports the same "policy:alpha" format as the default.
if [ -n "${POLICIES:-}" ]; then
    # `read -ra` takes ONE line. A multi-line POLICIES silently yielded an
    # empty array and the sweep then "completed" having run nothing.
    read -ra POLICIES <<< "$(echo $POLICIES)"
    [ ${#POLICIES[@]} -gt 0 ] || { echo "POLICIES is set but parsed to nothing" >&2; exit 1; }
else
    POLICIES=(
        # Baselines
        "round_robin"
        "least_loaded"
        # Carbon-aware α sweep (with SLA)
        # Spanning BOTH halves: 0.0 is carbon-only, 1.0 is latency-only.
        "carbon_aware:0.0"
        "carbon_aware:0.25"
        "carbon_aware:0.5"
        "carbon_aware:0.75"
        "carbon_aware:0.9"
        # SLA-aware carbon-optimal
        "ilp_carbon_sla"
        "iterative_ilp_carbon_sla"
    )
fi

# ── Derived paths ──
# Override to keep multiple sweeps organized:
#   RESULTS_ROOT=results/temporal/hour_18 bash scripts/sweep/sweep_all_policies.sh
RESULTS_ROOT="${RESULTS_ROOT:-results/sweep_$(date +%Y%m%d_%H%M)}"
mkdir -p "$RESULTS_ROOT"

# Save config
cat > "$RESULTS_ROOT/config.json" <<EOF
{
  "git_sha": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)",
  "seed": $SEED,
  "sim_datetime": "$SIM_DATETIME",
  "duration": $DURATION,
  "ttft_budget_ms": $TTFT_BUDGET,
  "tpot_budget_ms": $TPOT_BUDGET,
  "tpot_filter": "$TPOT_FILTER",
  "peer_scale": ${SCHED_PEER_SCALE:-$(echo $REGIONS | wc -w)},
  "bs_cap": $BS_CAP,
  "poll_mode": "$POLL_MODE",
  "poll_period_s": ${POLL_PERIOD_S:-1},
  "poll_wave_count": $POLL_WAVE_COUNT,
  "poll_wave_interval_s": $POLL_WAVE_INTERVAL_S,
  "model_name": "$MODEL_NAME",
  "tp_size": $TP_SIZE,
  "num_nodes": $NUM_NODES,
  "num_gpus": $NUM_GPUS,
  "max_batch_size": $MAX_BATCH_SIZE,
  "kv_capacity_source": "${KV_CAPACITY_SOURCE:-derived}",
  "pue_table": "${PUE_TABLE:-common/pue_by_dc.json}",
  "kv_queue_agg": "${KV_QUEUE_AGG:-mean}",
  "max_model_len": $MAX_MODEL_LEN,
  "osl_source": "$OSL_SOURCE",
  "ref_osl": $REF_OSL,
  "batch_window_ms": $BATCH_WINDOW_MS,
  "batch_window_size": $BATCH_WINDOW_SIZE,
  "loadgen_stream": ${LOADGEN_STREAM:-0},
  "loadgen_shards": ${LOADGEN_SHARDS:-1},
  "ilp_solve_process": ${ILP_SOLVE_PROCESS:-1},
  "arrival_rates": [$(IFS=,; echo "${ARRIVAL_RATES[*]}")],
  "policies": [$(printf '"%s",' "${POLICIES[@]}" | sed 's/,$//')]
}
EOF

echo "============================================================"
echo "  Sweep: ${#POLICIES[@]} policies × ${#ARRIVAL_RATES[@]} rates"
echo "  Duration: ${DURATION}s per run"
echo "  SLA: TTFT<${TTFT_BUDGET}ms  TPOT<${TPOT_BUDGET}ms"
echo "  Results: $RESULTS_ROOT"
echo "============================================================"

# ── OSL_SOURCE=predicted needs a trace that actually carries predictions ──
# Checked here rather than at serving time. The scheduler warns and falls back
# to the flat reference length for a request with no prediction, which silently
# turns the run into a third configuration that is neither of the two being
# compared -- and a sweep is hours long, so the place to find that out is
# before it starts. Only some traces have the column:
# all nine mixed_/mlperf_/long_<model>.csv traces do; the retired
# mixed_trace*.csv ones do not, and the predictor cannot be re-run on a trace
# that kept no prompt text.
if [ "$OSL_SOURCE" = "predicted" ]; then
    python3 - "$TRACE_CSV" <<'PYCHECK' || exit 1
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(f"OSL_SOURCE=predicted: {path} is empty")
if "predicted_osl" not in rows[0]:
    sys.exit(f"OSL_SOURCE=predicted: {path} has no predicted_osl column. "
             f"Use a trace that carries one (loadgen/scenarios/mixed_trace_with_osl.csv), "
             f"or run with OSL_SOURCE=oracle.")
bad = [r["sample_id"] for r in rows if int(r["predicted_osl"] or 0) <= 0]
if bad:
    sys.exit(f"OSL_SOURCE=predicted: {len(bad)} of {len(rows)} rows in {path} "
             f"have no predicted_osl (first: sample_id={bad[0]}). Every request "
             f"would fall back to the flat reference length, so the run would "
             f"measure neither source.")
print(f"  OSL check: all {len(rows)} rows of {path} carry a prediction")
PYCHECK
fi

# ── Generate experiment topology ──
# If TOPOLOGY_JSON is set and points to an existing file, use it as-is
# (for hardcoded spatial experiments). Otherwise generate from seed.
if [ -n "${TOPOLOGY_JSON:-}" ] && [ -f "$TOPOLOGY_JSON" ]; then
    echo "Using pre-built topology: $TOPOLOGY_JSON"
    # Skip the copy if source already lives at the destination path
    # (smoke_test_topology.sh and sweep_parallel.sh do this).
    SRC_REAL="$(realpath "$TOPOLOGY_JSON")"
    DST_REAL="$(realpath -m "$RESULTS_ROOT/topology.json")"
    if [ "$SRC_REAL" != "$DST_REAL" ]; then
        cp "$TOPOLOGY_JSON" "$RESULTS_ROOT/topology.json"
    fi
else
    echo "Generating topology (seed=$SEED)..."
    PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
        --seed "$SEED" --output "$RESULTS_ROOT/topology.json" 2>&1 | head -10
fi
# Every process launched below inherits this. common/network.py reads it at
# import and applies any "group_parents" split (several schedulers in one
# region); a topology without that key changes nothing.
export SIM_TOPOLOGY_FILE="$RESULTS_ROOT/topology.json"

# ── Helper functions ──
# POLICY_SLOT lets multiple parallel sweep instances coexist by offsetting
# all gRPC ports by SLOT * 1000. Slot 0 (default) keeps the original
# 50100/50200/50300 layout for backward compatibility.
POLICY_SLOT="${POLICY_SLOT:-0}"
DC_BASE_PORT=$(( 50100 + POLICY_SLOT * 1000 ))
SCHED_BASE_PORT=$(( 50200 + POLICY_SLOT * 1000 ))
PROV_PORT=$(( 50300 + POLICY_SLOT * 1000 ))

get_dc_port() {
    echo $(( DC_BASE_PORT + $1 ))
}

get_sched_port() {
    echo $(( SCHED_BASE_PORT + $1 ))
}

# Block until a server is actually accepting connections, instead of guessing
# with a fixed sleep. datacenter/server.py builds the Datacenter and all its
# engines BEFORE add_insecure_port/start, so an open port means the engine is
# built and serving -- not merely that the process exists.
wait_for_port() {
    local port="$1" name="$2" timeout_s="${3:-60}"
    local deadline=$(( $(date +%s) + timeout_s ))
    until (: < "/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "  ✗ ABORT: $name never opened port $port (waited ${timeout_s}s)" >&2
            return 1
        fi
        sleep 0.1
    done
}

wait_for_port_free() {
    # The mirror of wait_for_port, and it exists because the sweep only ever
    # checked that a port OPENS. Launching a sweep while the previous one is
    # still tearing down meant a server hit "Failed to bind to address" and
    # died, and the only symptom was "never opened port 50202 (waited 60s)" --
    # which reads as a slow start, not as a port someone else was holding. By
    # the time anyone looked, the port was free and nothing was running.
    #
    # This stays a CONNECT test on purpose. It answers "is someone still
    # serving here", which is the case worth aborting on. It cannot see a
    # socket in TIME_WAIT, which refuses connections while still blocking
    # bind() -- but making it see one would only turn a situation the server
    # recovers from into a 45 s abort. TIME_WAIT is handled where it belongs,
    # in the bind itself: common/grpc_bind.py retries for 60 s, in every
    # server that binds a port.
    local port="$1" name="$2" timeout_s="${3:-45}"
    local deadline=$(( $(date +%s) + timeout_s ))
    while (: < "/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "  ✗ ABORT: port $port ($name) is still held after ${timeout_s}s." >&2
            echo "     Another sweep is probably still running or tearing down." >&2
            echo "     Check with: ss -ltnp | grep $port" >&2
            return 1
        fi
        sleep 0.2
    done
}

start_provisioner() {
    local logfile="$1/provisioner.log"
    PYTHONPATH=proto:. \
    GRPC_PORT=$PROV_PORT \
    POLL_PERIOD_S=${POLL_PERIOD_S:-1} \
    POLL_MODE="$POLL_MODE" \
    POLL_WAVE_COUNT="$POLL_WAVE_COUNT" \
    POLL_WAVE_INTERVAL_S="$POLL_WAVE_INTERVAL_S" \
    RTT_BETA=0.5 \
    RTT_NORM="${RTT_NORM:-cap}" \
    RTT_HEAT_CAP="${RTT_HEAT_CAP:-32}" \
    STATIC_ENDPOINTS_JSON="$ENDPOINT_JSON" \
    PROVISIONER_TIMESERIES="$1/fleet_timeseries.csv" \
    VERBOSITY=$VERBOSITY \
    python -m provisioner.server > "$logfile" 2>&1 &
    echo $!
}

# gRPC pools are sized per ROLE, because the two roles hold threads for very
# different reasons and one number for both is what broke a 2400 req/s run.
#
# A datacentre holds one thread per request it is SERVING, bounded by its own
# admission. A scheduler holds one per request it is WAITING ON -- and a
# forwarded request occupies TWO, one in the origin waiting for the remote and
# one in the remote waiting for the datacentre. On this topology round_robin
# forwards about three quarters of everything, so a scheduler needs roughly
# 1.75 threads per in-flight request where a datacentre needs one.
#
# Measured when both were pinned at 50000: the four schedulers each grew to
# exactly 50146 and the sixteen datacentres sat at 2555. Four times fifty
# thousand is 200000, which was the whole task budget, so the datacentres could
# not get threads and the run deadlocked with a load average of 0.4 -- nothing
# running, everything blocked. The pools also never shrink, so one burst during
# ramp-up holds the budget for the rest of the run.
#
# The defaults here cover 6000 req/s on the 512-engine fleet: about 21000
# in flight, so ~37000 scheduler threads across four and ~21000 datacentre
# threads across sixteen.
start_dc() {
    local dc_id="$1" port="$2" ci="$3" logdir="$4"
    local logfile="$logdir/dc_${dc_id}.log"
    # PUE is deliberately NOT passed. The site reads its own from
    # common/pue_by_dc.json (common/pue.py), keyed by dc_id, so every process
    # that needs it -- the datacentre and scripts/sweep/collect_energy.py -- reads
    # the same table. Exporting PUE here would override the table for EVERY
    # site at once, which is the opposite of per-site. To vary one site, edit
    # the table; to sweep a uniform value, export DEFAULT_PUE.
    # One BLOCKED THREAD per in-flight request, at both layers: the datacentre
    # handler waits on the engine and the scheduler waits on the datacentre.
    # The default of 5000 is a tipping point, not a ceiling you approach
    # gracefully -- measured on least_loaded at 1200 req/s, the SAME setting
    # gave 442.9 g in 152.7 s with no errors in one sweep and 2489.0 g in
    # 2108 s with 15.3% DEADLINE_EXCEEDED in the next. Whether a transient
    # concentration crosses 5000 decides which, and nothing in the output says
    # the limit was hit: the wait happens before the handler runs, so queue_ms
    # and service_ms both stay near zero while wall clock goes to 33 minutes.
    # At 50000 the same run is clean (442.3 g, 150.5 s, 0 errors).
    #
    # Raising it needs the OS limits raised too -- the testbed runs
    # vm.max_map_count=524288 (in /etc/sysctl.d/99-loadgen.conf), which allows
    # 262144 threads per process against the 65530 default's 32765.
    PYTHONPATH=proto:. \
    DC_ID="$dc_id" \
    GRPC_MAX_WORKERS="${GRPC_MAX_WORKERS_DC:-${GRPC_MAX_WORKERS:-8000}}" \
    GRPC_PORT="$port" \
    NUM_NODES=$NUM_NODES \
    NUM_GPUS_PER_NODE=$NUM_GPUS \
    TP_SIZE=$TP_SIZE \
    MAX_BATCH_SIZE=$MAX_BATCH_SIZE \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    KV_CAPACITY_SOURCE="${KV_CAPACITY_SOURCE:-derived}" \
    KV_QUEUE_AGG="${KV_QUEUE_AGG:-mean}" \
    OSL_SOURCE="$OSL_SOURCE" \
    REF_OSL="$REF_OSL" \
    LATENCY_MODE=$DC_LATENCY_MODE \
    MODEL_NAME=$MODEL_NAME \
    LATENCY_CSV="$LATENCY_CSV" \
    CARBON_INTENSITY="$ci" \
    BS_PROFILE_DIR="${BS_PROFILE_DIR:-}" \
    VERBOSITY=$DC_VERBOSITY \
    python -m datacenter.server > "$logfile" 2>&1 &
    echo $!
}

start_scheduler() {
    local sched_id="$1" group="$2" port="$3" policy="$4" alpha="$5" logdir="$6" tag="$7"
    local logfile="$logdir/sched_${sched_id}.log"

    # One decision log per scheduler, PER POLICY. The path used to be derived
    # from the global $DECISION_LOG and $sched_id alone, so every policy in a
    # sweep wrote the same four files: the last one to run kept its decisions
    # and the other thirteen were overwritten. Nothing reported it, because a
    # sweep that produces four well-formed logs looks exactly like a sweep that
    # produces fifty-six. Anchoring to $logdir -- which is already per-policy --
    # is what makes the file name mean what it says. Only the basename of
    # $DECISION_LOG is honoured; a directory in it would put every policy's log
    # back in one place, which is the bug.
    local declog=""
    if [[ -n "${DECISION_LOG:-}" ]]; then
        local dl_base="${DECISION_LOG##*/}"
        declog="$logdir/${dl_base%.jsonl}_${sched_id}.jsonl"
    fi
    # One BLOCKED THREAD per in-flight request, at both layers: the datacentre
    # handler waits on the engine and the scheduler waits on the datacentre.
    # The default of 5000 is a tipping point, not a ceiling you approach
    # gracefully -- measured on least_loaded at 1200 req/s, the SAME setting
    # gave 442.9 g in 152.7 s with no errors in one sweep and 2489.0 g in
    # 2108 s with 15.3% DEADLINE_EXCEEDED in the next. Whether a transient
    # concentration crosses 5000 decides which, and nothing in the output says
    # the limit was hit: the wait happens before the handler runs, so queue_ms
    # and service_ms both stay near zero while wall clock goes to 33 minutes.
    # At 50000 the same run is clean (442.3 g, 150.5 s, 0 errors).
    #
    # Raising it needs the OS limits raised too -- the testbed runs
    # vm.max_map_count=524288 (in /etc/sysctl.d/99-loadgen.conf), which allows
    # 262144 threads per process against the 65530 default's 32765.
    PYTHONPATH=proto:. \
    SCHED_ID="$sched_id" \
    SCHEDULER_GROUP="$group" \
    GRPC_MAX_WORKERS="${GRPC_MAX_WORKERS_SCHED:-${GRPC_MAX_WORKERS:-16000}}" \
    GRPC_PORT="$port" \
    ROUTING_POLICY="$policy" \
    ROUTING_ALPHA="$alpha" \
    SIMULATE_RTT=$SIMULATE_RTT \
    DECISION_LOG="$declog" \
    DECISION_LOG_CANDIDATES="${DECISION_LOG_CANDIDATES:-}" \
    TTFT_BUDGET_MS=$TTFT_BUDGET \
    TPOT_BUDGET_MS=$TPOT_BUDGET \
    TPOT_FILTER="$TPOT_FILTER" \
    SCHED_PEER_SCALE="${SCHED_PEER_SCALE:-$(echo $REGIONS | wc -w)}" \
    BS_CAP=$BS_CAP \
    MAX_BATCH_SIZE=$MAX_BATCH_SIZE \
    OSL_SOURCE=$OSL_SOURCE \
    REF_OSL=$REF_OSL \
    MODEL_NAME=$MODEL_NAME \
    TP_SIZE=$TP_SIZE \
    BATCH_WINDOW_MS=$BATCH_WINDOW_MS \
    BATCH_WINDOW_SIZE=$BATCH_WINDOW_SIZE \
    GIL_SWITCH_INTERVAL_MS="${GIL_SWITCH_INTERVAL_MS:-}" \
    ILP_MAX_ITERATIONS=$ILP_MAX_ITER \
    STATIC_ENDPOINTS_JSON="$ENDPOINT_JSON" \
    VERBOSITY=$VERBOSITY \
    LOG_TAG="$tag" \
    python -m scheduler.server > "$logfile" 2>&1 &
    echo $!
}

run_loadgen() {
    # LOADGEN_VERBOSITY=2 also writes results/per_request/requests_*.csv, one
    # file per shard per run, incrementally. Those columns are a SUBSET of the
    # run's own results.csv, so the only thing they add is data from a run that
    # was killed before results.csv was written -- and they are never pruned:
    # 294 files and 7.1 GB by 2026-09-06, on a disk at 83%. Default 1.
    local rate="$1" output_csv="$2" logdir="$3" sched_addrs="$4"
    local logfile="$logdir/loadgen.log"
    # --concurrent 0 lets client.py size the pool itself, from rate x duration
    # capped at 30000. It used to be pinned at 5000 here, which quietly undid
    # the safeguard client.py documents at length: a pool narrower than the
    # offered load turns an OPEN-loop generator into a CLOSED-loop one.
    #
    # Measured, at 1600 req/s with a 3.77 s round trip, 5000 workers ceilings
    # the client at 1326/s. The run delivered 1219/s, its pool_wait grew
    # monotonically from 1.2 s to 35.4 s -- an unbounded queue, never steady --
    # and 120 s of arrivals stretched to 157 s. Worse than the lost rate: the
    # ceiling is pool/wall, so a policy with higher latency is handed LESS
    # load and rescues itself. The carbon policies are exactly the high-latency
    # ones, so the bias runs straight through the comparison the paper makes.
    #
    # The wide pool is not expensive: ThreadPoolExecutor creates threads on
    # demand, so only the peak in-flight count is ever real.
    #
    # LOADGEN_SHARDS splits the generator across processes. One generator is a
    # single asyncio loop on a single thread, so it saturates a core: measured
    # on the eight-group fleet, it offered 4183 req/s against a requested 5000
    # with pool_wait p50 at 8.5 s, which is the closed-loop failure above by
    # another route. Each shard drives a disjoint set of schedulers, takes its
    # share of the rate, and writes its own CSV; the shards are merged below so
    # everything downstream sees one file. The generator listens on no port, so
    # the shards cannot collide -- and the schedulers still discover and dial
    # each other from the full endpoint list, unchanged.
    local shards="${LOADGEN_SHARDS:-1}"
    if [ "$shards" -le 1 ]; then
        PYTHONPATH=proto:. \
        SCHED_ADDRS="$sched_addrs" \
        python -m loadgen.client \
            --duration "$DURATION" \
            --arrival-distribution poisson \
            --arrival-rate "$rate" \
            --trace-csv "$TRACE_CSV" \
            --output "$output_csv" \
            --concurrent "${LOADGEN_CONCURRENT:-0}" \
            --verbosity "${LOADGEN_VERBOSITY:-1}" \
            > "$logfile" 2>&1
        return
    fi

    local per_shard
    per_shard=$(awk -v r="$rate" -v n="$shards" 'BEGIN{printf "%.4f", r/n}')
    local pids=() parts=()
    local i
    for ((i = 0; i < shards; i++)); do
        local out="${output_csv%.csv}_s${i}.csv"
        parts+=("$out")
        PYTHONPATH=proto:. \
        SCHED_ADDRS="$sched_addrs" \
        python -m loadgen.client \
            --duration "$DURATION" \
            --arrival-distribution poisson \
            --arrival-rate "$per_shard" \
            --trace-csv "$TRACE_CSV" \
            --output "$out" \
            --shard "${i}/${shards}" \
            --concurrent "${LOADGEN_CONCURRENT:-0}" \
            --verbosity "${LOADGEN_VERBOSITY:-1}" \
            > "${logfile%.log}_s${i}.log" 2>&1 &
        pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i"; done

    # One file downstream. The header comes from the first shard that produced
    # one; a shard that died leaves no file and its absence shows up as a
    # shortfall in the generator check, which is where it belongs.
    : > "$output_csv"
    local wrote_header=0
    for out in "${parts[@]}"; do
        [ -s "$out" ] || continue
        if [ "$wrote_header" -eq 0 ]; then
            cat "$out" >> "$output_csv"
            wrote_header=1
        else
            tail -n +2 "$out" >> "$output_csv"
        fi
    done
    cat "${logfile%.log}"_s*.log > "$logfile" 2>/dev/null || true
}

cleanup_pids() {
    # Two-phase shutdown:
    #   1. SIGTERM for graceful exit (gives gRPC servers a chance to flush logs)
    #   2. After CLEANUP_GRACE_SEC, SIGKILL anything still alive
    # We do NOT use bare `wait` because Python gRPC servers don't always
    # honor SIGTERM (they block in server.wait_for_termination), which would
    # hang the script forever.
    local grace="${CLEANUP_GRACE_SEC:-5}"
    for pid in "$@"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    local deadline=$(($(date +%s) + grace))
    for pid in "$@"; do
        while kill -0 "$pid" 2>/dev/null; do
            if [ "$(date +%s)" -ge "$deadline" ]; then
                kill -KILL "$pid" 2>/dev/null || true
                break
            fi
            sleep 0.1
        done
    done
}

# ── Read topology ──
REGIONS=$(PYTHONPATH=proto:. python3 -c "
import json
topo = json.load(open('$RESULTS_ROOT/topology.json'))
for region in topo['topology']:
    print(region)
")

# ── Main sweep loop ──
for policy_spec in "${POLICIES[@]}"; do
    # Parse policy:alpha format
    if [[ "$policy_spec" == *:* ]]; then
        POLICY="${policy_spec%%:*}"
        ALPHA="${policy_spec##*:}"
        TAG="${POLICY}_a${ALPHA}"
    else
        POLICY="$policy_spec"
        ALPHA="0.5"
        TAG="${POLICY}"
    fi

    for RATE in "${ARRIVAL_RATES[@]}"; do
        RUN_DIR="$RESULTS_ROOT/${TAG}/rate_${RATE}"
        mkdir -p "$RUN_DIR/logs"

        echo ""
        echo "── Policy=$POLICY  α=$ALPHA  Rate=$RATE req/s ──"

        PIDS=()

        # Generate static endpoints JSON for this run
        ENDPOINTS_FILE="$RUN_DIR/endpoints.json"
        DC_IDX=0
        SCHED_IDX=0
        PYTHONPATH=proto:. python3 -c "
import json, sys
from datetime import datetime
sys.path.insert(0, '.')
from common.carbon import CarbonLoader
topo = json.load(open('$RESULTS_ROOT/topology.json'))
all_dcs = topo['all_dcs']
dt = datetime.fromisoformat('$SIM_DATETIME')
# The CI file is per year, so the snapshot's year picks it.
loader = CarbonLoader('carbon_intensity', year=dt.year, regions=all_dcs)

endpoints = {}
dc_idx = 0
for dc in all_dcs:
    port = $DC_BASE_PORT + dc_idx
    endpoints[dc] = f'localhost:{port}'
    dc_idx += 1

sched_idx = 0
for region in topo['topology']:
    port = $SCHED_BASE_PORT + sched_idx
    endpoints[f'sched-{region}'] = f'localhost:{port}'
    sched_idx += 1

endpoints['provisioner'] = f'localhost:$PROV_PORT'

# Also output DC carbon intensities
ci_map = {}
for dc in all_dcs:
    ci_map[dc] = loader.get_ci(dc, dt)

json.dump({'endpoints': endpoints, 'carbon': ci_map}, sys.stdout, indent=2)
" > "$ENDPOINTS_FILE"

        # Extract endpoints and CI
        ENDPOINT_JSON=$(PYTHONPATH=proto:. python3 -c "
import json
data = json.load(open('$ENDPOINTS_FILE'))
print(json.dumps(data['endpoints']))
")

        # Start DCs FIRST. The provisioner used to start ahead of them, so its
        # opening poll wave hit ports nobody was listening on yet: every run's
        # provisioner.log begins with "Poll <dc> failed: _InactiveRpcError", and
        # a site that missed the wave stayed invisible to the schedulers until
        # the next discovery refresh -- 10 s of a 120 s run, or the whole run if
        # it never registered at all.
        #
        # Use process substitution (`done < <(cmd)`) instead of
        # `cmd | while read` so the loop body runs in the parent shell — that
        # way PIDS+=() persists. Also capture each DC's PID via command
        # substitution so cleanup_pids can later kill it; the previous
        # `> /dev/null` discarded those PIDs and orphaned the DC processes.
        DC_IDX=0
        DC_PORTS=()
        DC_NAMES=()
        while read dc_name ci; do
            PORT=$(get_dc_port $DC_IDX)
            wait_for_port_free "$PORT" "DC $dc_name" || {
                cleanup_pids "${PIDS[@]}"; exit 1; }
            DC_PID=$(start_dc "$dc_name" "$PORT" "$ci" "$RUN_DIR/logs")
            PIDS+=($DC_PID)
            DC_PORTS+=($PORT)
            DC_NAMES+=("$dc_name")
            DC_IDX=$((DC_IDX + 1))
        done < <(PYTHONPATH=proto:. python3 -c "
import json
data = json.load(open('$ENDPOINTS_FILE'))
for dc, ci in data['carbon'].items():
    print(f'{dc} {ci}')
")

        # Readiness, not a guess. A DC that needs longer than 2 s to build its
        # engines used to be dispatched to before it could answer.
        for i in "${!DC_PORTS[@]}"; do
            wait_for_port "${DC_PORTS[$i]}" "DC ${DC_NAMES[$i]}" 120 || {
                cleanup_pids "${PIDS[@]}"; exit 2; }
        done

        # Only now the provisioner: every site it polls is already serving.
        STATIC_ENDPOINTS_JSON="$ENDPOINT_JSON" \
        wait_for_port_free "$PROV_PORT" "provisioner" || {
            cleanup_pids "${PIDS[@]}"; exit 1; }
        PROV_PID=$(start_provisioner "$RUN_DIR/logs")
        PIDS+=($PROV_PID)
        # Verify it actually came up. The datacentres and the schedulers were
        # already checked this way; the provisioner was not, and that gap cost
        # a whole grid cell.
        #
        # 2026-09-07, grid_v3, carbon_aware:0.5 at 1200 req/s: the port was
        # free by the check above, the provisioner then hit "Failed to bind to
        # address [::]:50300" anyway, and NOTHING NOTICED. With no provisioner
        # there are no snapshots, so every scheduler ran on an empty dc_view
        # for the whole run and fell through to the no-view branch in
        # server.py -- which picked the alphabetically first datacentre. 100%
        # of the load went to AU_VIC, one of the DIRTIEST sites in the fleet,
        # and the cell reported zero errors. It was caught only because the
        # run took 622 s instead of 220. A shorter outage would have produced
        # a plausible-looking carbon number and nobody would have known.
        wait_for_port "$PROV_PORT" "provisioner" 150 || {
            echo "     Without a provisioner no snapshot is ever pushed, so" >&2
            echo "     every policy would run blind on an empty view. See" >&2
            echo "     $RUN_DIR/logs/provisioner.log" >&2
            # Kill what this cell started before leaving. This abort used to
            # exit with every datacentre still serving, and on 2026-09-10 the
            # next two sweeps in the chain both died on "port 50100 is still
            # held" before their first cell.
            cleanup_pids "${PIDS[@]}"
            exit 1
        }

        # Start schedulers. Same fix: capture the PID instead of discarding it.
        SCHED_IDX=0
        SCHED_PORTS=()
        SCHED_NAMES=()
        for REGION in $REGIONS; do
            PORT=$(get_sched_port $SCHED_IDX)
            SCHED_ID="sched-${REGION}"
            wait_for_port_free "$PORT" "scheduler $SCHED_ID" || {
                cleanup_pids "${PIDS[@]}"; exit 1; }
            SCHED_PID=$(STATIC_ENDPOINTS_JSON="$ENDPOINT_JSON" \
                start_scheduler "$SCHED_ID" "$REGION" "$PORT" "$POLICY" "$ALPHA" "$RUN_DIR/logs" "$TAG")
            PIDS+=($SCHED_PID)
            SCHED_PORTS+=($PORT)
            SCHED_NAMES+=("$SCHED_ID")
            SCHED_IDX=$((SCHED_IDX + 1))
        done

        for i in "${!SCHED_PORTS[@]}"; do
            wait_for_port "${SCHED_PORTS[$i]}" "scheduler ${SCHED_NAMES[$i]}" 150 || {
                cleanup_pids "${PIDS[@]}"; exit 2; }
        done

        # One more settle: the schedulers are serving, but their dc_view is
        # empty until the provisioner has polled every site and pushed once.
        # The old flat `sleep 2; sleep 3` happened to cover a 2 s cycle, but it
        # was a constant -- raise POLL_PERIOD_S and the first requests would
        # land on schedulers that had never seen a snapshot. Deriving it from
        # the period keeps that true at any setting.
        sleep "$(awk -v p="${POLL_PERIOD_S:-1}" 'BEGIN{print 2*p}')"

        # Run loadgen
        OUTPUT_CSV="$RUN_DIR/results.csv"
        echo "  Running loadgen: rate=$RATE duration=$DURATION → $OUTPUT_CSV"
        STATIC_ENDPOINTS_JSON="$ENDPOINT_JSON" \
        run_loadgen "$RATE" "$OUTPUT_CSV" "$RUN_DIR/logs" ""

        # Did the GENERATOR keep up? A pool narrower than the offered load
        # does not error and does not drop anything -- it just sends late, so
        # the row count looks perfect while the experiment quietly stops being
        # open-loop. That cost a full 14-policy run on 2026-09-04, caught only
        # by reading pool_wait afterwards. The ceiling is pool/wall, so the
        # SLOWEST policy is handed the LEAST load and flatters itself; the bias
        # runs straight through the comparison. Warn per policy rather than
        # abort: a late run is still readable if you know it was late.
        python3 - "$OUTPUT_CSV" "$DURATION" "$RATE" <<'PYCHK'
import csv, statistics as st, sys
try:
    every = list(csv.DictReader(open(sys.argv[1])))
except OSError:
    sys.exit(0)
if not every:
    sys.exit(0)
# OFFERED load counts every request that went out, errors included. Dividing
# only the successful ones by the span conflates two unrelated failures: a
# generator that could not send, and a fleet that answered with an error. On
# frontier_full it called iterative_ilp@800 "558/s of 800" when the generator
# had sent all 95792 on schedule and 32% came back DEADLINE_EXCEEDED.
rows = [r for r in every if not (r.get("error") or "").strip()]
pw = [float(r.get("pool_wait_ms") or 0) for r in rows] or [0.0]
snd = [float(r["send_s"]) for r in every]
span = max(snd) - min(snd)
n_err = len(every) - len(rows)
p50, p99 = st.median(pw), sorted(pw)[int(0.99 * (len(pw) - 1))]
eff = len(every) / span if span > 0 else 0
want = float(sys.argv[3])
# The test is DELIVERED RATE, not pool_wait. pool_wait is a symptom, and it
# has a blind spot that let a biased run through: on probe_rate3 at 800 req/s,
# ilp_carbon_sla sat at p50 215 ms -- under the old 250 ms threshold, so it
# printed "generator ok" -- while being handed 732/s against round_robin's
# 798/s. Comparing a policy's carbon at 92% of another's load is the exact
# bias this check exists to catch, and the symptom test missed it.
#
# The coupling is closed-loop: in-flight = rate x latency, so a policy whose
# latency grows fills the pool and is then handed LESS work, which flatters it.
# The generator itself is not the ceiling -- round_robin delivered 1199/s at
# the same pool size in the same sweep.
shortfall = 1.0 - (eff / want) if want > 0 else 0.0
# The delivered rate is ONE canonical line, printed unconditionally, before
# any verdict. It used to live inside a branch, and adding the error report
# captured the elif/else that followed it -- so a run WITH errors printed no
# rate at all, which is the case where you most need to know whether the
# generator also fell short. Verdicts qualify this line; they never replace it.
print(f"  generator: delivered {eff:.0f}/s of {want:.0f}/s, "
      f"pool_wait p50 {p50:.0f} ms, p99 {p99:.0f} ms")
if shortfall > 0.05:
    print(f"  !! GENERATOR BOUND: {100 * shortfall:.0f}% short. Sent {len(every)} "
          f"over {span:.0f}s against {sys.argv[2]}s requested.")
    print(f"     The run is CLOSED-loop and NOT comparable against a policy that "
          f"got its full rate.")
elif p50 > 250:
    print(f"     queueing without losing load -- the rate held, the arrivals "
          f"were late.")
if n_err:
    pct = 100.0 * n_err / len(every)
    tag = "  !! " if pct > 1.0 else "  "
    print(f"{tag}errors: {n_err} of {len(every)} ({pct:.1f}%) came back failed. "
          f"A run losing requests is not comparable either -- and it is a "
          f"DIFFERENT fault from the generator falling behind.")
PYCHK

        # Pre-flight sanity check: abort sweep if >50% of requests errored out.
        # Catches crashed batch threads, dead DCs, etc. — avoids wasting a full
        # overnight sweep on broken runs.
        ERR_ABORT_THRESHOLD="${ERR_ABORT_THRESHOLD:-50}"
        READ_BACK=$(python3 -c "
import csv, sys
total = err = 0
try:
    with open('$OUTPUT_CSV') as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get('status') != 'OK':
                err += 1
except FileNotFoundError:
    print('NO_CSV 0 0'); sys.exit(0)
pct = (100.0 * err / total) if total else 0.0
print(f'OK {total} {err} {pct:.1f}')
")
        read STATUS TOTAL ERRS ERR_PCT <<< "$READ_BACK"
        if [ "$STATUS" = "NO_CSV" ]; then
            echo "  ✗ ABORT: results.csv not produced for $POLICY rate=$RATE"
            # Kill ONLY this slot's own processes (pkill -f would race other parallel slots)
            cleanup_pids "${PIDS[@]}"
            exit 2
        fi
        echo "  Requests: total=$TOTAL errors=$ERRS (${ERR_PCT}%)"
        if python3 -c "import sys; sys.exit(0 if float('$ERR_PCT') > float('$ERR_ABORT_THRESHOLD') else 1)"; then
            echo "  ✗ ABORT: error rate ${ERR_PCT}% exceeds threshold ${ERR_ABORT_THRESHOLD}%"
            echo "  Check scheduler logs in: $RUN_DIR/logs/"
            cleanup_pids "${PIDS[@]}"
            exit 3
        fi

        # Collect energy (must run BEFORE killing processes)
        echo "  Collecting energy..."
        # Write endpoints to a file for collect_energy.py
        PYTHONPATH=proto:. python3 -c "
import json
data = json.load(open('$ENDPOINTS_FILE'))
with open('$RUN_DIR/static_endpoints.json', 'w') as f:
    json.dump(data['endpoints'], f, indent=2)
"
        # SIM_DURATION_S is deliberately NOT set. collect_energy.py derives the
        # window from the loadgen CSV (max recv - min send), which is what the
        # fleet was actually up for; pinning it to the configured DURATION
        # billed a 485 s run over 120 s. Measured on probe_rate2 at 1200 req/s
        # the under-count reached 4.0x, and correcting it INVERTED the ranking:
        # ilp_carbon_sla went from 143.3 g (better than round_robin's 428.3) to
        # 578.7 g (worse than its corrected 524.8). The policies that stretch
        # the run the most were the ones the error flattered most.
        #
        # --requests-csv is what makes that derivation possible; without it the
        # collector falls back to 0 and every site reports no energy at all.
        PYTHONPATH=proto:. \
        STATIC_ENDPOINTS_FILE="$RUN_DIR/static_endpoints.json" \
        python scripts/sweep/collect_energy.py \
            "$RUN_DIR/energy.json" \
            --requests-csv "$OUTPUT_CSV" \
            2>&1 | tee -a "$RUN_DIR/logs/energy.log" || true

        # Cleanup all processes — kill ONLY this slot's own PIDs.
        # `pkill -f "python -m datacenter.server"` would match every parallel
        # slot's DC processes too, racing them and breaking energy collection
        # for any slot that hasn't yet reached this point.
        echo "  Cleaning up..."
        cleanup_pids "${PIDS[@]}"
        sleep 2

        echo "  Done: $OUTPUT_CSV"
    done
done

echo ""
echo "============================================================"
echo "  Sweep complete! Results in: $RESULTS_ROOT"
echo "============================================================"

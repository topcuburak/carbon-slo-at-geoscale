#!/usr/bin/env bash
#
# The end-to-end run: every registered policy, one topology, one arrival rate.
#
#   scripts/sweep/run_e2e_all_policies.sh
#   RATE=120 DURATION=180 scripts/sweep/run_e2e_all_policies.sh
#
# Topology: 4 scheduler groups x 4 datacentres x 4 nodes, 8 cards per node at
# TP=1 -- 8 engines per node, 32 per site, 512 in the fleet, and a capacity of
# roughly 7200 req/s. The four groups are chosen for spread rather than
# convenience: at the sampled hour the fleet runs 37 to 449 gCO2/kWh, a 12x
# range, with the greenest sites in south_america and europe_north and the
# dirtiest in asia and us_east. A carbon policy therefore has a real reason to
# cross a region, and a real RTT to pay for it.
#
# DURATION is 200 s, not 120, because of what the transient costs. Warm-up and
# drain are not the operating point being measured and they are a different
# size for every policy -- at 1500 req/s on the smaller fleet the drain was 20%
# of a round_robin run and 55% of a carbon_aware one. Lengthening the arrival
# window does not shorten the transient, it dilutes it: the same absolute
# warm-up and drain fall from 20-55% of a run to 12-33%. It does NOT rescue a
# policy whose throughput is below the offered rate; there the queue grows
# without bound and no duration helps. `logs/fleet_timeseries.csv` records the
# per-epoch fleet load so the steady-state window can be found afterwards.
#
# WHY 1600 req/s. The MLPerf trace is short work -- 841 prompt tokens and 71
# output tokens on average -- so a request holds a seat for ~3.3 s against the
# ~8.3 s of the mixed trace this experiment used to run. Occupancy is
# rate x service / seats, and at 200 req/s this fleet would sit at 3% of its
# seats: nothing queues, no site is ever unattractive, the SLA filter excludes
# nobody, every policy meets every budget, and the comparison collapses to
# carbon alone. 1600 req/s puts the fleet at 21.5%, which is where the run that
# actually produced a differentiated frontier sat (20.3%). Fleet occupancy is
# the wrong number to read on its own, though: the carbon policies concentrate
# about 28% of the load on one site, so the busiest datacentre runs near 48%.
#
# The cost is the arrival process. The load generator's arrivals stop being
# Poisson somewhere past 200 req/s, and 1600 is 80% of the ~2000 where its
# worker pool binds. Less bursty arrivals mean LESS queueing than a true
# Poisson stream would produce, uniformly across policies, so it flatters every
# arm equally rather than reordering them -- state it as a limitation. Check it
# rather than assume it: results.csv should hold ~192000 rows (rate x duration),
# and a short count means the generator, not the fleet, was the bottleneck.
#
# Every policy writes its own directory:
#
#   $RESULTS_ROOT/<policy>/rate_<rate>/
#       results.csv     one row per request
#       energy.json     per-datacentre joules and carbon, after FinalizeEnergy
#       logs/           datacentre, scheduler and provisioner logs, and
#                       decisions_sched-*.jsonl when DECISION_LOG is set
#
# On the two SLOs. MLPerf defines Llama-3.1-8B Server as TTFT <= 2 s /
# TPOT <= 100 ms and Interactive as TTFT <= 0.5 s / TPOT <= 30 ms.
#
#   TTFT 3000 ms   MLPerf's 2000 plus what OUR system-under-test has and a
#                  single-system SUT does not: a cross-region hop and a remote
#                  scheduler. Measured on this topology, forwarding round trip
#                  p99 is 635-700 ms and ILP scheduling p99 is 257 ms, so
#                  2000 + 700 + 257 = 2957. Derived, not chosen.
#   TPOT  100 ms   MLPerf Server. It does NOT bind here and is not meant to:
#                  at TP=1 the pure decode step is 51 ms at the KV-limited
#                  maximum batch of 213, or 60-67 ms once prefill interference
#                  is counted, so no site is ever excluded on TPOT. The
#                  Interactive threshold of 30 ms binds from batch ~64 up --
#                  run TPOT_BUDGET=30 TTFT_BUDGET=500 for that scenario.
#
# THE TRACE CYCLES, and at this rate it cycles hard. MLPerf's CNN/DailyMail
# split is 13368 samples -- that is the spec, not a shortage -- and 1600 req/s
# for 120 s dispatches 192000 requests, so the trace is traversed 14.4 times.
# MLPerf's own Server scenario cycles its dataset too, so this is practice
# rather than deviation, but two things follow. The repetition period is 8.35 s,
# which would matter if the file had burst structure; measured, it does not --
# ISL means across deciles span 5.4% and lag-1 autocorrelation is -0.013, so
# the order is shuffled and the load carries no periodic signal. What does
# follow is that the number of DISTINCT samples is 13368, not 192000: quote
# that as the effective sample size for anything driven by request content.
#
# Overriding: every knob above the topology check is `${VAR:-default}`, so a
# different shape is an environment away rather than an edit.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RATE="${RATE:-1600}"
DURATION="${DURATION:-200}"

# Every knob below is overridable, because the shape of this experiment is the
# experiment. They used to be literals in the env list at the bottom, which
# silently ignored anything exported from outside.
MODEL_NAME="${MODEL_NAME:-llama3_8B}"
TP_SIZE="${TP_SIZE:-1}"
NUM_NODES="${NUM_NODES:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
N_DCS_PER="${N_DCS_PER:-4}"
# Which scheduler groups the fleet spans. One scheduler process per group, and
# a group is a zero-RTT domain, so this is a modelling choice AND the only
# parallelism the scheduler layer has: its event loop is single-threaded, so
# four groups means four cores no matter how many the host has.
#
# Measured at 4000 req/s on the four-group fleet: each scheduler process sat at
# 123% CPU (the loop pinned at ~100% plus gRPC's C threads) while the sixteen
# datacentre processes idled at 18% each and 57 of the testbed's 64 cores went unused.
# Delivered throughput stopped at ~2600 req/s there and would not move for any
# offered rate up to 4760.
#
# common/network.py defines eight groups and an RTT matrix for all 36 pairs, so
# REGIONS="... 8 names ..." with N_DCS_PER=2 gives the same 16 datacentres and
# the same 512 engines across eight scheduler processes instead of four.
REGIONS="${REGIONS:-europe_north us_east asia south_america}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TTFT_BUDGET="${TTFT_BUDGET:-3000}"
TPOT_BUDGET="${TPOT_BUDGET:-100}"
TRACE_CSV="${TRACE_CSV:-loadgen/scenarios/mlperf_llama3_8B.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-results/e2e_$(date +%Y%m%d)}"
TOPOLOGY="${TOPOLOGY:-$RESULTS_ROOT/topology/topology.json}"

# Every registered policy. carbon_aware's alpha traces its carbon-vs-latency
# frontier, so it gets a spread rather than one value -- and the spread has to
# cover BOTH halves. alpha is the LATENCY weight:
#     score = alpha * norm_latency + (1 - alpha) * norm_carbon,  minimised
# so 1.0 ignores carbon and 0.0 ignores latency. Sweeping 0.5-0.95, as this
# did at first, is the latency-dominant half only; measured, every value in it
# served 49-51% of requests on the clean sites, indistinguishable from
# round_robin.
# One line, space separated: sweep_all_policies.sh parses this with `read -ra`,
# which reads a single line -- a multi-line list silently becomes zero policies
# and the sweep runs nothing while reporting success.
POLICIES="${POLICIES:-round_robin round_robin_local least_loaded least_loaded_local weighted_least_loaded carbon_aware:0.0 carbon_aware:0.25 carbon_aware:0.5 carbon_aware:0.75 carbon_aware:0.9 carbon_aware_local:0.9 ilp_carbon_sla iterative_ilp_carbon_sla facility_ilp_carbon_sla}"

if [[ ! -f "$TOPOLOGY" ]]; then
    echo "  generating topology -> $TOPOLOGY"
    mkdir -p "$(dirname "$TOPOLOGY")"
    PYTHONPATH=proto:. python3 scripts/sweep/generate_experiment_config.py \
        --regions $REGIONS \
        --n-dcs-per "$N_DCS_PER" --n-nodes-per-dc "$NUM_NODES" --seed 42 --output "$TOPOLOGY"
fi

echo "=== end-to-end: $(echo $POLICIES | wc -w) policies at ${RATE} req/s for ${DURATION}s"
echo "    topology $TOPOLOGY"
echo "    results  $RESULTS_ROOT"

# Run inside a scope of our own, with room for the threads this needs.
#
# The chain is synchronous: a request in flight parks a thread in the load
# generator, in each scheduler that handles it, and in the datacentre. A policy
# that CONCENTRATES load makes the queues at the chosen sites hold far more at
# once, so thread count tracks concentration, not request rate. Two runs have
# died on it with `RuntimeError: can't start new thread`, and the failure is
# silent -- the scheduler stops answering, the client's 2000 s timeout keeps
# waiting, and the sweep hangs with every datacentre empty.
#
# The limit is systemd's per-scope TasksMax, not the stack size and not
# vm.max_map_count -- both were measured and neither moved the ceiling. Inside
# VS Code's snap scope it is DefaultTasksMax=18409, which caps a process at
# ~17.9k threads. A scope of our own reaches 38k, measured, without root.
# user.slice allows 40500, so that is the ceiling on this machine.
#
# systemd-run is TESTED before being exec'd. `exec` replaces this shell, so a
# systemd-run that exists but cannot work -- no user D-Bus session, which is
# the normal state of a plain `ssh host <command>` -- took the whole run down
# with it. Being present on PATH is not the same as being usable.
if [[ -z "${SIM_IN_SCOPE:-}" ]] && command -v systemd-run >/dev/null 2>&1; then
    if systemd-run --user --scope --quiet -- true >/dev/null 2>&1; then
        export SIM_IN_SCOPE=1
        exec systemd-run --user --scope -p TasksMax="${SIM_TASKS_MAX:-40000}" --quiet -- \
            bash "$0" "$@"
    fi
    _cg=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null)
    _max=$(cat "/sys/fs/cgroup${_cg}/pids.max" 2>/dev/null || echo unknown)
    echo "  WARNING: systemd-run --user is present but not usable here, so this" >&2
    echo "  run gets no scope of its own. The thread ceiling is whatever this" >&2
    echo "  session already has: pids.max=${_max}. A concentrating policy can" >&2
    echo "  exhaust it and the failure is SILENT -- the scheduler stops" >&2
    echo "  answering and the sweep hangs. Over ssh, 'ssh -t' or" >&2
    echo "  'loginctl enable-linger \$USER' gets the session back." >&2
fi

TOPOLOGY_JSON="$TOPOLOGY" \
RESULTS_ROOT="$RESULTS_ROOT" \
POLICIES="$POLICIES" \
ARRIVAL_RATES="$RATE" \
DURATION="$DURATION" \
SEED=42 \
MODEL_NAME=$MODEL_NAME \
TP_SIZE=$TP_SIZE \
NUM_NODES=$NUM_NODES \
NUM_GPUS=$NUM_GPUS \
MAX_BATCH_SIZE=$MAX_BATCH_SIZE \
MAX_MODEL_LEN=$MAX_MODEL_LEN \
TTFT_BUDGET=$TTFT_BUDGET \
TPOT_BUDGET=$TPOT_BUDGET \
TPOT_FILTER="${TPOT_FILTER:-arrival}" \
BS_CAP="${BS_CAP:-0}" \
POLL_MODE="${POLL_MODE:-sync}" \
POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}" \
POLL_PERIOD_S="${POLL_PERIOD_S:-2}" \
BATCH_WINDOW_MS="${BATCH_WINDOW_MS:-100}" \
BATCH_WINDOW_SIZE="${BATCH_WINDOW_SIZE:-300}" \
ILP_MAX_ITER=3 \
OSL_SOURCE=oracle \
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}" \
TRACE_CSV="$TRACE_CSV" \
VERBOSITY=0 \
DC_VERBOSITY=0 \
bash scripts/sweep/sweep_all_policies.sh

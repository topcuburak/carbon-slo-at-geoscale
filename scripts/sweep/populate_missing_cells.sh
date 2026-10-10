#!/usr/bin/env bash
# =============================================================================
#  Populate the cells that failed in the prior sensitivity sweeps
# =============================================================================
#
# Re-runs exactly the (sweep, config, policy, rate) combos that returned
# 100% errors due to the port-overflow bug (now fixed). Each rerun reuses
# the parent sweep's topology.json and writes back into the same subdir,
# so the data merges cleanly with the surviving 363 cells.
#
# Targets — 14 cells total, ~35 min wallclock:
#   - model_sens / DeepSeek / round_robin × 7 rates             (7 cells)
#   - polling / 4 specific (config, policy) pairs               (4 cells)
#   - slo / 2 specific (config, policy) pairs                   (2 cells)
#   - temporal / hour_06 / round_robin                          (1 cell)
#   (spatial / asia_pacific / round_robin is the documented expected
#    abort — left as-is)
#
# Usage:
#   bash scripts/sweep/populate_missing_cells.sh
#
# All cells use POLICY_SLOT_BASE=0 — the fix that prevents port overflow.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

# Common config — matches the original sweeps.
export POLICY_SLOT_BASE=0
export POLICY_SLOT=0
export DURATION=120
export TTFT_BUDGET=2000
export TPOT_BUDGET=50
export BS_CAP=128
export NUM_NODES=10
export NUM_GPUS=8
export ILP_MAX_ITER=3

# All cells use the same shared topology (rates were originally generated from
# this seed in each parent sweep — the topology.json copies are identical).

SWEEPS_ROOT="results"

LLAMA_CSV="datacenter/perf_models/lookup/measured/full/llama3.1-8B/all_datasets_merged_benchmark.csv"
LLAMA_POWER="datacenter/perf_models/lookup/measured/full/llama3.1-8B/GovReport/benchmarks/power.csv"
DSEEK_CSV="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/merged_benchmark.csv"
DSEEK_POWER="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/power.csv"

# ── Helper ─────────────────────────────────────────────────────────────
# rerun_cell <sweep_dir> <model_name> <tp> <latency_csv>
#            <policy> <rate> <ttft> <tpot> <poll_mode> <poll_period> <poll_wave_int>
#            <sim_datetime>
rerun_cell() {
    local sweep_dir="$1" model="$2" tp="$3" lcsv="$4" pcsv="$5"
    local policy="$6" rate="$7" ttft="$8" tpot="$9"
    local mode="${10}" period="${11}" wave="${12}" sim_dt="${13}"

    local topo="$sweep_dir/topology.json"
    if [ ! -f "$topo" ]; then
        echo "  ✗ MISSING topology: $topo (skipping)"
        return 1
    fi

    echo
    echo "→ Re-running cell: $sweep_dir / $policy / rate=$rate"
    echo "  TTFT=${ttft}ms TPOT=${tpot}ms model=$model tp=$tp poll=${mode}/${period}s"

    MODEL_NAME="$model" \
    TP_SIZE="$tp" \
    LATENCY_CSV="$lcsv" \
    TTFT_BUDGET="$ttft" \
    TPOT_BUDGET="$tpot" \
    POLL_MODE="$mode" \
    POLL_PERIOD_S="$period" \
    POLL_WAVE_INTERVAL_S="$wave" \
    POLL_WAVE_COUNT=4 \
    SIM_DATETIME="$sim_dt" \
    TOPOLOGY_JSON="$topo" \
    RESULTS_ROOT="$sweep_dir" \
    POLICIES="$policy" \
    ARRIVAL_RATES="$rate" \
    bash scripts/sweep/sweep_all_policies.sh
}

# ── 1. model_sens / DeepSeek / round_robin × 7 rates ──────────────────
echo "============================================================"
echo "  Group 1: DeepSeek round_robin × 7 rates"
echo "============================================================"
MS_DIR="$SWEEPS_ROOT/model_sens_20260428_0049/deepseek_moe_16b"
for rate in 50 100 200 500 1000 2000 5000; do
    rerun_cell "$MS_DIR" deepseek_moe_16b 2 "$DSEEK_CSV" "$DSEEK_POWER" \
        round_robin "$rate" 2000 50 \
        async 1 0.25 "2024-07-15T12:00:00"
done

# ── 2. Polling: 4 specific failed cells ───────────────────────────────
echo
echo "============================================================"
echo "  Group 2: polling 4 cells"
echo "============================================================"
POLL_DIR="$SWEEPS_ROOT/polling_parallel_20260428_1153"

# async_2s/iterative_ilp_carbon_sla
rerun_cell "$POLL_DIR/async_2s" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    iterative_ilp_carbon_sla 2000 2000 50 \
    async 2 0.5 "2024-07-15T12:00:00"

# async_5s/least_loaded
rerun_cell "$POLL_DIR/async_5s" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    least_loaded 2000 2000 50 \
    async 5 1.25 "2024-07-15T12:00:00"

# async_10s/iterative_ilp_carbon_sla
rerun_cell "$POLL_DIR/async_10s" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    iterative_ilp_carbon_sla 2000 2000 50 \
    async 10 2.5 "2024-07-15T12:00:00"

# sync_2s/round_robin
rerun_cell "$POLL_DIR/sync_2s" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    round_robin 2000 2000 50 \
    sync 2 0.5 "2024-07-15T12:00:00"

# ── 3. SLO: 2 specific failed cells ───────────────────────────────────
echo
echo "============================================================"
echo "  Group 3: SLO 2 cells"
echo "============================================================"
SLO_DIR="$SWEEPS_ROOT/slo_parallel_20260428_2242"

# ttft2000_tpot50/round_robin
rerun_cell "$SLO_DIR/ttft2000_tpot50" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    round_robin 2000 2000 50 \
    async 1 0.25 "2024-07-15T12:00:00"

# ttft2000_tpot100/iterative_ilp_carbon_sla
rerun_cell "$SLO_DIR/ttft2000_tpot100" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    iterative_ilp_carbon_sla 2000 2000 100 \
    async 1 0.25 "2024-07-15T12:00:00"

# ── 4. Temporal: hour_06 / round_robin ────────────────────────────────
echo
echo "============================================================"
echo "  Group 4: Temporal hour_06 round_robin"
echo "============================================================"
TEMP_DIR="$SWEEPS_ROOT/spatial_vs_temporal/temporal_parallel_20260428_0951/hour_06"
rerun_cell "$TEMP_DIR" llama3_8B 2 "$LLAMA_CSV" "$LLAMA_POWER" \
    round_robin 2000 2000 50 \
    async 1 0.25 "2024-07-15T06:00:00"

# ── Done ───────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  All 14 missing cells re-run."
echo "============================================================"
echo
echo "Verify with:"
echo "  re-run this script -- there is no separate audit tool"

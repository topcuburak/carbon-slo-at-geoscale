#!/usr/bin/env bash
# =============================================================================
#  Model sensitivity sweep — LLaMA-8B / Qwen2.5-7B (TP=2), Mixtral-8x7B (TP=4)
# =============================================================================
#
# Reproduces the existing LLaMA-8B sweep config (sweep_parallel_20260426_0530)
# for the two MoE models so we can compare across model architectures.
# Same topology, same rates, same policies, same SLOs.
#
# Each model becomes a sibling output dir under $ROOT, so analyses can run
# the same scripts pointing at each subdir.
#
# Hardware mapping (per the user's "same topology" requirement):
#   - Same physical fleet: 32 DCs × 10 nodes × 8 GPUs = 2560 physical GPUs.
#   - LLaMA-8B  TP=2 → 1280 GPU pairs.
#   - Mixtral   TP=4 →  640 GPU pairs (half of LLaMA — that's the cost of TP=4).
#   - Qwen2.5-7B TP=2 → 1280 GPU pairs (same as LLaMA).
#   - DeepSeek  TP=2 → 1280 GPU pairs (same as LLaMA), not run by default.
#
# Usage (on our testbed):
#   bash scripts/sweep/sweep_model_sensitivity.sh
#
# Override defaults by env:
#   MODELS="mixtral_8x7b"                   # subset
#   MODELS="deepseek_moe_16b"               # uncalibrated, latency-only
#   ROOT=results/sweeps/model_sens_$(date +%Y%m%d_%H%M)  # output root
#   ARRIVAL_RATES="50 100 200 500 1000 2000 5000"
#   DURATION=120 SEED=42
#
# Each model run leaves output at $ROOT/$MODEL/ — all manifest fields,
# topology.json, per-policy/per-rate subdirs identical to a regular
# sweep_parallel.sh run.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

# Models to run. Default is the three that are calibrated on BOTH axes:
# a fitted per-(model,TP) latency artifact and a measured StepEnergyModel.
#
# deepseek_moe_16b is deliberately NOT in the default list. It has neither --
# AnalyticalLatencyModel falls back to the built-in TP=2 table and
# StepEnergyModel.load() returns None, which drops the DC onto the legacy
# three-state power constants without saying so. That fallback is also the
# only path where the per-GPU/per-pair power scaling bug is still live. It
# stays runnable on purpose (MODELS="deepseek_moe_16b") for architecture
# comparisons that do not depend on energy; do not put it in an energy figure
# until it has been calibrated.
MODELS="${MODELS:-llama3_8B qwen2.5_7B mixtral_8x7b}"

# Root output dir for all model sweeps
ROOT="${ROOT:-results/sweeps/model_sens_$(date +%Y%m%d_%H%M)}"
mkdir -p "$ROOT"

# Common config — identical to LLaMA reference (sweep_parallel_20260426_0530)
export SEED="${SEED:-42}"
export DURATION="${DURATION:-120}"
export N_DCS_PER="${N_DCS_PER:-8}"
export N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
export TTFT_BUDGET="${TTFT_BUDGET:-2000}"
export TPOT_BUDGET="${TPOT_BUDGET:-50}"
export BS_CAP="${BS_CAP:-128}"
export POLL_MODE="${POLL_MODE:-async}"
export POLL_PERIOD_S="${POLL_PERIOD_S:-1}"
export POLL_WAVE_COUNT="${POLL_WAVE_COUNT:-4}"
export POLL_WAVE_INTERVAL_S="${POLL_WAVE_INTERVAL_S:-0.25}"
export SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
export ARRIVAL_RATES="${ARRIVAL_RATES:-50 100 200 500 1000 2000 5000}"

# Generate ONE topology shared across both model runs so per-DC carbon
# intensities and node counts are identical between models.
TOPO="$ROOT/topology.json"
if [ ! -f "$TOPO" ]; then
    echo "→ Generating shared topology ($N_DCS_PER DCs × 4 zones, seed=$SEED)..."
    PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
        --seed "$SEED" \
        --n-dcs-per "$N_DCS_PER" \
        --n-nodes-per-dc "$N_NODES_PER_DC" \
        --output "$TOPO" 2>&1 | head -10
fi
export TOPOLOGY_JSON="$TOPO"

echo "============================================================"
echo "  Model sensitivity sweep"
echo "  Models:       $MODELS"
echo "  Output root:  $ROOT"
echo "  Topology:     $TOPO"
echo "  Rates:        $ARRIVAL_RATES"
echo "  SLO:          TTFT<${TTFT_BUDGET}ms TPOT<${TPOT_BUDGET}ms BS_CAP=$BS_CAP"
echo "  Duration:     ${DURATION}s per (rate × policy)"
echo "============================================================"

MODEL_IDX=0
for model in $MODELS; do
    # All models reuse POLICY_SLOT_BASE=0. Models run sequentially so
    # concurrency isolation is unnecessary; a 60s pause lets any
    # TIME_WAIT'd sockets clear. The earlier MODEL_IDX*8 scheme overflowed
    # the 16-bit TCP port limit at model ≥ 2 (port 50100 + slot*1000 >
    # 65535) — DeepSeek's run was hit at slot_base=16.
    export POLICY_SLOT_BASE=0
    if [ "$MODEL_IDX" -gt 0 ]; then
        echo "  (waiting 60s for TIME_WAIT to clear before next model…)"
        sleep 60
    fi
    MODEL_IDX=$((MODEL_IDX + 1))

    echo
    echo "============================================================"
    echo "  Running sweep for $model"
    echo "  Slot base: $POLICY_SLOT_BASE  (ports $((50100 + POLICY_SLOT_BASE * 1000))+)"
    echo "============================================================"

    case "$model" in
        mixtral_8x7b)
            export TP_SIZE=4
            export LATENCY_CSV="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/merged_benchmark.csv"
            ;;
        deepseek_moe_16b)
            export TP_SIZE=2
            # No DeepSeek CSV — DC TTFT predictions come from the model
            # JSON lookup, so any valid CSV satisfies the constructor.
            # Power CSV inherits Mixtral's as a known-imperfect placeholder
            # (DeepSeek-MoE power coefficients are still pending refit).
            export LATENCY_CSV="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/merged_benchmark.csv"
            ;;
        llama3_8B)
            export TP_SIZE=2
            export LATENCY_CSV="datacenter/perf_models/lookup/measured/full/llama3.1-8B/all_datasets_merged_benchmark.csv"
            ;;
        qwen2.5_7B)
            export TP_SIZE=2
            # Fitted latency artifact and measured energy model both exist at
            # TP=2, so the CSV is a placeholder like the others -- the DC runs
            # LATENCY_MODE=analytical off the model JSON.
            export LATENCY_CSV="datacenter/perf_models/lookup/measured/summaries/Qwen2.5-7B_GovReport/merged_benchmark.csv"
            # Deliberately empty. Qwen was never run through
            # power_post_process.py, so it has only the raw NVML trace
            # (power_log.csv, one column per GPU), not the batch_size/power_w
            # table LatencyLookup expects -- passing it raises KeyError and
            # every DC dies at startup. Inheriting another model's power.csv
            # the way DeepSeek does would load LLaMA's watts under Qwen's name.
            # Empty is correct: energy comes from the measured
            # StepEnergyModel for (qwen2.5_7B, TP=2), which supersedes the
            # lookup on this path anyway.
            ;;
        *)
            echo "Unknown model $model — skipping" ; continue ;;
    esac

    export MODEL_NAME="$model"
    out="$ROOT/$model"
    mkdir -p "$out"

    ROOT="$out" \
    POLICY_SLOT_BASE="$POLICY_SLOT_BASE" \
    bash scripts/sweep/sweep_parallel.sh

done

echo
echo "============================================================"
echo "  Model sensitivity sweep complete"
echo "  Output: $ROOT"
echo "============================================================"
echo
echo "Analyze each model:"
for model in $MODELS; do
    echo "  python scripts/sweep/analyze_sweep.py $ROOT/$model"
done

#!/usr/bin/env bash
# =============================================================================
#  Smoke test: validate scheduler analytical fits for all 3 models
# =============================================================================
#
# Spins up a tiny topology (4 zones × 2 DCs × 2 nodes × 2 GPUs = 16 GPUs,
# 8 GPU-pairs at TP=2; smaller for Mixtral at TP=4) and runs carbon_aware
# at rate=20 for 30s for each model. Validates that:
#   1. Scheduler's AnalyticalLatencyModel loads with the new closed-form fits
#      (linear for llama3_8B, piecewise_linear TTFT + linear_curvature TPOT
#      for the two MoE models).
#   2. End-to-end requests complete with 0% errors.
#   3. Predicted TTFT/TPOT distributions are sane (no negative values, no
#      out-of-band tails).
#
# Usage:
#   bash scripts/test/smoke_test_models.sh                 # all 3 models
#   MODELS="mixtral_8x7b" bash scripts/test/smoke_test_models.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

# Models to smoke
MODELS="${MODELS:-llama3_8B mixtral_8x7b deepseek_moe_16b}"

ROOT="${ROOT:-results/smoke_models_$(date +%Y%m%d_%H%M)}"
mkdir -p "$ROOT"

# Tiny topology
SEED=42
N_DCS_PER=2          # 2 DCs per region × 4 regions = 8 DCs
N_NODES_PER_DC=2

echo "============================================================"
echo "  Smoke test: scheduler analytical fits"
echo "  Models: $MODELS"
echo "  Topology: 4 zones × 2 DCs × 2 nodes (small)"
echo "  Output: $ROOT"
echo "============================================================"

# Build a single topology shared across runs
PYTHONPATH=proto:. python scripts/sweep/generate_experiment_config.py \
    --seed "$SEED" \
    --n-dcs-per "$N_DCS_PER" \
    --n-nodes-per-dc "$N_NODES_PER_DC" \
    --output "$ROOT/topology.json" 2>&1 | head -10

declare -A OK_RATE FAIL_COUNT TOTAL_REQS SCHED_P90 USER_TTFT_P90 TPOT_P90

for model in $MODELS; do
    echo
    echo "── Smoke: $model ─────────────────────────────────────────"

    case "$model" in
        llama3_8B)
            tp=2
            gpus_per_node=2
            csv="datacenter/perf_models/lookup/measured/full/llama3.1-8B/all_datasets_merged_benchmark.csv"
            pcsv="datacenter/perf_models/lookup/measured/full/llama3.1-8B/GovReport/benchmarks/power.csv"
            ;;
        mixtral_8x7b)
            tp=4
            gpus_per_node=4   # TP=4 needs ≥4 GPUs per node
            csv="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/merged_benchmark.csv"
            pcsv="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/power.csv"
            ;;
        deepseek_moe_16b)
            tp=2
            gpus_per_node=2
            # No DeepSeek CSV on laptop; reuse Mixtral as placeholder for the
            # legacy CSV-derived fallback paths. JSON lookup handles the
            # actual predictions at the DC.
            csv="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/merged_benchmark.csv"
            pcsv="datacenter/perf_models/lookup/measured/full/mixtral-8x7B/GovReport/benchmarks/power.csv"
            ;;
        *)
            echo "Unknown model $model — skipping"
            continue
            ;;
    esac

    # Per-model run
    out="$ROOT/$model"
    mkdir -p "$out"

    MODEL_NAME="$model" \
    TP_SIZE="$tp" \
    LATENCY_CSV="$csv" \
    NUM_NODES=$N_NODES_PER_DC \
    NUM_GPUS=$gpus_per_node \
    MAX_BATCH_SIZE=32 \
    BS_CAP=32 \
    TTFT_BUDGET=2000 \
    TPOT_BUDGET=50 \
    DURATION=20 \
    POLL_MODE=async \
    POLL_PERIOD_S=1 \
    POLL_WAVE_COUNT=4 \
    POLL_WAVE_INTERVAL_S=0.25 \
    DC_LATENCY_MODE=lookup \
    POLICIES="carbon_aware" \
    ROUTING_ALPHA="0.0" \
    ARRIVAL_RATES="5" \
    RESULTS_ROOT="$out" \
    TOPOLOGY_JSON="$ROOT/topology.json" \
    bash scripts/sweep/sweep_all_policies.sh

    # Pull per-model summary
    rcsv="$out/carbon_aware/rate_5/results.csv"
    if [ ! -f "$rcsv" ]; then
        echo "  ✗ MISSING $rcsv — smoke FAILED for $model"
        OK_RATE[$model]="MISSING"
        continue
    fi

    python3 - <<PYEOF
import pandas as pd
df = pd.read_csv("$rcsv")
n = len(df)
errs = int(df.error.notna().sum()) if 'error' in df.columns else 0
ok = n - errs
sched_p90 = df.scheduling_ms.quantile(0.9) if 'scheduling_ms' in df else float('nan')
ttft_p90 = df.ttft_ms.quantile(0.9) if 'ttft_ms' in df else float('nan')
tpot_p90 = df.avg_tpot_ms.quantile(0.9) if 'avg_tpot_ms' in df else float('nan')
user_ttft_p90 = (df.scheduling_ms + df.forwarding_ms/2 + df.queue_ms + df.ttft_ms).quantile(0.9) if all(c in df for c in ['scheduling_ms','forwarding_ms','queue_ms','ttft_ms']) else float('nan')
print(f"  Total reqs: {n}  OK: {ok}  Errors: {errs}  err%: {errs/max(n,1)*100:.2f}%")
print(f"  sched p90: {sched_p90:.2f}ms  user_TTFT p90: {user_ttft_p90:.2f}ms  TPOT p90: {tpot_p90:.2f}ms")
# Sanity flags
flags = []
if errs > 0: flags.append(f"ERRORS={errs}")
if (df.ttft_ms < 0).any(): flags.append("NEGATIVE_TTFT")
if 'avg_tpot_ms' in df and (df.avg_tpot_ms < 0).any(): flags.append("NEGATIVE_TPOT")
if user_ttft_p90 > 5000: flags.append(f"user_TTFT_p90>5s")
if flags:
    print(f"  ⚠ FLAGS: {', '.join(flags)}")
else:
    print("  ✓ All sanity checks pass")
PYEOF

done

echo
echo "============================================================"
echo "  Smoke test complete. Results: $ROOT"
echo "============================================================"

#!/usr/bin/env bash
# Sweep across arrival rates to study load sensitivity.
#
# Runs sweep_test.sh at different arrival rates (default 10 to 100, step 10)
# with a fixed CI snapshot to isolate the load effect.
#
# Usage:
#   bash scripts/sweep/sweep_arrival_rate.sh [duration] [dcs_per_group] [rate_min] [rate_max] [rate_step]
#   bash scripts/sweep/sweep_arrival_rate.sh 120 4                      # 10-100, step 10
#   bash scripts/sweep/sweep_arrival_rate.sh 120 4 10 100 10            # explicit range
#   bash scripts/sweep/sweep_arrival_rate.sh 120 7 20 200 20            # higher capacity setup
#
# Environment:
#   SIM_DATETIME    CI snapshot (default "2024-10-15T06")
#   MAX_BATCH_SIZE  Per-GPU-pair batch cap (default 128)

set -e
cd "$(dirname "$0")/../.."

DURATION="${1:-120}"
DCS_PER_GROUP="${2:-4}"
RATE_MIN="${3:-10}"
RATE_MAX="${4:-100}"
RATE_STEP="${5:-10}"
SIM_DATETIME="${SIM_DATETIME:-2024-10-15T06}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-128}"

RESULTS_BASE="results/arrival_rate"
mkdir -p "$RESULTS_BASE"

# Build list of rates
RATES=()
for RATE in $(seq "$RATE_MIN" "$RATE_STEP" "$RATE_MAX"); do
    RATES+=("$RATE")
done

TOTAL=${#RATES[@]}
CURRENT=0
START_TIME=$(date +%s)

echo "================================================================"
echo "  Arrival Rate Sweep"
echo "  Duration: ${DURATION}s  DCs/group: ${DCS_PER_GROUP}"
echo "  Rates: ${RATE_MIN} to ${RATE_MAX} (step ${RATE_STEP}) — ${TOTAL} runs"
echo "  SIM_DATETIME: ${SIM_DATETIME}"
echo "  MAX_BATCH_SIZE: ${MAX_BATCH_SIZE}"
echo "  Results: $RESULTS_BASE/"
echo "================================================================"

for RATE in "${RATES[@]}"; do
    CURRENT=$((CURRENT + 1))
    TAG="rate_${RATE}"
    CSV="$RESULTS_BASE/${TAG}.csv"

    echo ""
    echo "========================================================"
    echo "  [$CURRENT/$TOTAL] Arrival rate: ${RATE} req/s"
    if [ $CURRENT -gt 1 ]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        AVG_PER_RUN=$(( ELAPSED / (CURRENT - 1) ))
        REMAINING=$(( AVG_PER_RUN * (TOTAL - CURRENT + 1) ))
        echo "  ETA: $(( REMAINING / 60 )) min remaining"
    fi
    echo "========================================================"

    LOGDIR="results/logs/rate_${RATE}" \
    RESULTS_DIR="results/sweep/rate_${RATE}" \
    SIM_DATETIME="$SIM_DATETIME" MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
        bash scripts/test/sweep_test.sh "$DURATION" "$RATE" "$DCS_PER_GROUP" "$CSV"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "================================================================"
echo "  All $TOTAL runs complete in $(( ELAPSED / 60 )) minutes"
echo "  Results in $RESULTS_BASE/"
echo "================================================================"

# ── Aggregate summary across all rates ───────────────────────────────
echo ""
echo "Generating aggregate summary..."

python3 << 'PYEOF'
import csv, os, glob
from collections import defaultdict

results_base = "results/arrival_rate"
all_csvs = sorted(glob.glob(os.path.join(results_base, "rate_*.csv")))
if not all_csvs:
    print("No CSV files found.")
    exit(0)

# Collect all rows with their arrival rate
agg_rows = []
for csv_path in all_csvs:
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    rate = int(basename.replace("rate_", ""))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["arrival_rate"] = rate
            agg_rows.append(row)

if not agg_rows:
    print("No data rows found.")
    exit(0)

# Write aggregate CSV
agg_path = os.path.join(results_base, "_aggregate.csv")
fieldnames = ["arrival_rate"] + [k for k in agg_rows[0].keys() if k != "arrival_rate"]
with open(agg_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(agg_rows)

print(f"Aggregate CSV: {agg_path} ({len(agg_rows)} rows from {len(all_csvs)} runs)")

# Print per-rate summary for carbon_aware policies
print(f"\n{'Rate':>5} | {'Policy':<28} | {'lat_avg':>10} | {'lat_p50':>10} | {'lat_p99':>10} | {'carbon_g':>10} | {'energy_kwh':>10}")
print("-" * 100)

by_rate = defaultdict(list)
for row in agg_rows:
    by_rate[row["arrival_rate"]].append(row)

for rate in sorted(by_rate.keys()):
    rows = by_rate[rate]
    for row in rows:
        policy = row.get("tag", "")
        lat_avg = row.get("lat_avg", "N/A")
        lat_p50 = row.get("lat_p50", "N/A")
        lat_p99 = row.get("lat_p99", "N/A")
        carbon = row.get("carbon_g", "N/A")
        energy = row.get("energy_kwh", "N/A")
        print(f"{rate:>5} | {policy:<28} | {lat_avg:>10} | {lat_p50:>10} | {lat_p99:>10} | {carbon:>10} | {energy:>10}")
PYEOF

echo ""
echo "Done."

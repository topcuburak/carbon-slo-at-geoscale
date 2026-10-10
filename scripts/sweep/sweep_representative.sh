#!/usr/bin/env bash
# Run sweep_test.sh across representative hours for robust Pareto analysis.
#
# 1. Diurnal: all 24 hours of July 15, 2024 (00:00–23:00 UTC)
# 2. Seasonal: Jan 15, Apr 15, Jul 15, Oct 15 at 14:00 UTC
#    (Jul 15 T14 already covered in diurnal → 27 unique runs)
#
# Usage:
#   bash scripts/sweep/sweep_representative.sh [duration] [arrival_rate] [dcs_per_group]
#   bash scripts/sweep/sweep_representative.sh 120 50 4
#
# Results go to results/representative/<datetime>/summary.csv

set -e
cd "$(dirname "$0")/../.."

DURATION="${1:-120}"
ARRIVAL_RATE="${2:-50}"
DCS_PER_GROUP="${3:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-128}"

RESULTS_BASE="results/representative"
mkdir -p "$RESULTS_BASE"

# Track progress
TOTAL=27
CURRENT=0
START_TIME=$(date +%s)

run_one() {
    local SIM_DT="$1"
    local TAG="$2"
    local CSV="$RESULTS_BASE/${TAG}.csv"

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================================"
    echo "  [$CURRENT/$TOTAL] $TAG  (SIM_DATETIME=$SIM_DT)"
    if [ $CURRENT -gt 1 ]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        AVG_PER_RUN=$(( ELAPSED / (CURRENT - 1) ))
        REMAINING=$(( AVG_PER_RUN * (TOTAL - CURRENT + 1) ))
        echo "  ETA: $(( REMAINING / 60 )) min remaining"
    fi
    echo "========================================================"

    LOGDIR="results/logs/${TAG}" \
    RESULTS_DIR="results/sweep/${TAG}" \
    SIM_DATETIME="$SIM_DT" MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
        bash scripts/test/sweep_test.sh "$DURATION" "$ARRIVAL_RATE" "$DCS_PER_GROUP" "$CSV"
}

echo "================================================================"
echo "  Representative Sweep"
echo "  Duration: ${DURATION}s  Rate: ${ARRIVAL_RATE} req/s  DCs/group: ${DCS_PER_GROUP}"
echo "  MAX_BATCH_SIZE: ${MAX_BATCH_SIZE}"
echo "  Total runs: $TOTAL"
echo "  Results: $RESULTS_BASE/"
echo "================================================================"

# ── Diurnal: 24 hours of July 15, 2024 ──────────────────────────────
for HOUR in $(seq -w 0 23); do
    run_one "2024-07-15T${HOUR}" "july15_h${HOUR}"
done

# ── Seasonal: 4 dates at 14:00 UTC (skip Jul 15 T14, already done) ──
run_one "2024-01-15T14" "jan15_h14"
run_one "2024-04-15T14" "apr15_h14"
# july15_h14 already covered above
run_one "2024-10-15T14" "oct15_h14"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "================================================================"
echo "  All $TOTAL runs complete in $(( ELAPSED / 60 )) minutes"
echo "  Results in $RESULTS_BASE/"
echo "================================================================"

# ── Aggregate summary across all runs ────────────────────────────────
echo ""
echo "Generating aggregate summary..."

python3 << 'PYEOF'
import csv, os, glob

results_base = "results/representative"
all_csvs = sorted(glob.glob(os.path.join(results_base, "*.csv")))
if not all_csvs:
    print("No CSV files found.")
    exit(0)

# Collect all rows with their source datetime tag
agg_rows = []
for csv_path in all_csvs:
    tag = os.path.splitext(os.path.basename(csv_path))[0]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["datetime_tag"] = tag
            agg_rows.append(row)

if not agg_rows:
    print("No data rows found.")
    exit(0)

# Write aggregate CSV
agg_path = os.path.join(results_base, "_aggregate.csv")
fieldnames = ["datetime_tag"] + [k for k in agg_rows[0].keys() if k != "datetime_tag"]
with open(agg_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(agg_rows)

print(f"Aggregate CSV: {agg_path} ({len(agg_rows)} rows from {len(all_csvs)} runs)")

# Print per-policy stats across all datetime snapshots
from collections import defaultdict
by_policy = defaultdict(list)
for row in agg_rows:
    key = row.get("tag", row.get("policy", "unknown"))
    by_policy[key].append(row)

print(f"\n{'Policy':<28} | {'Runs':>4} | {'lat_avg mean':>12} | {'lat_avg std':>11} | {'carbon_g mean':>13} | {'carbon_g std':>12}")
print("-" * 95)
for policy in sorted(by_policy.keys()):
    rows = by_policy[policy]
    lat_vals = [float(r["lat_avg"]) for r in rows if r.get("lat_avg")]
    carbon_vals = [float(r["carbon_g"]) for r in rows if r.get("carbon_g")]
    n = len(lat_vals)
    if n == 0:
        continue
    lat_mean = sum(lat_vals) / n
    lat_std = (sum((v - lat_mean)**2 for v in lat_vals) / n) ** 0.5
    if carbon_vals:
        c_mean = sum(carbon_vals) / len(carbon_vals)
        c_std = (sum((v - c_mean)**2 for v in carbon_vals) / len(carbon_vals)) ** 0.5
        print(f"{policy:<28} | {n:>4} | {lat_mean:>12.1f} | {lat_std:>11.1f} | {c_mean:>13.1f} | {c_std:>12.1f}")
    else:
        print(f"{policy:<28} | {n:>4} | {lat_mean:>12.1f} | {lat_std:>11.1f} | {'N/A':>13} | {'N/A':>12}")
PYEOF

echo ""
echo "Done."

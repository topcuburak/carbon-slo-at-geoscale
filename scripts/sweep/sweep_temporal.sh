#!/usr/bin/env bash
# =============================================================================
#  Temporal CI sweep — same DCs, different hours of the day
# =============================================================================
#
# Runs the full policy × rate sweep at multiple wall-clock timestamps to
# study how carbon savings depend on the time of day. Each timestamp gets
# its own results subdirectory.
#
# Usage:
#   bash scripts/sweep/sweep_temporal.sh
#
# Environment overrides:
#   HOURS="0 6 12 18"          subset of hours to sweep (UTC)
#   SIM_DATE="2024-07-15"      ISO date to anchor the sweep
#   SEED=42                    DC topology seed (held fixed by default)
#   ARRIVAL_RATES="30"         arrival rates (one rate keeps it tractable)
#   ERR_ABORT_THRESHOLD=95     pre-flight error threshold (per inner sweep)
#   ROOT=results/temporal      where to put per-hour subdirs
# =============================================================================

set -euo pipefail

HOURS="${HOURS:-0 3 6 9 12 15 18 21}"
SIM_DATE="${SIM_DATE:-2024-07-15}"
SEED="${SEED:-42}"
ARRIVAL_RATES="${ARRIVAL_RATES:-30}"
ERR_ABORT_THRESHOLD="${ERR_ABORT_THRESHOLD:-95}"
ROOT="${ROOT:-results/temporal_$(date +%Y%m%d_%H%M)}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  Temporal sweep"
echo "  Date:        $SIM_DATE"
echo "  Hours:       $HOURS"
echo "  Seed:        $SEED"
echo "  Rates:       $ARRIVAL_RATES"
echo "  Output root: $ROOT"
echo "============================================================"

# Save outer-sweep manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "temporal",
  "date": "$SIM_DATE",
  "hours": [$(echo $HOURS | sed 's/ /, /g')],
  "seed": $SEED,
  "arrival_rates": "$ARRIVAL_RATES",
  "started_at": "$(date -Iseconds)"
}
EOF

for HOUR in $HOURS; do
    HOUR_PADDED=$(printf "%02d" "$HOUR")
    SIM_DATETIME="${SIM_DATE}T${HOUR_PADDED}:00:00"
    SUBDIR="$ROOT/hour_${HOUR_PADDED}"

    echo ""
    echo "============================================================"
    echo "  → Hour ${HOUR_PADDED}:00 UTC  ($SIM_DATETIME)"
    echo "  → Output: $SUBDIR"
    echo "============================================================"

    SEED="$SEED" \
    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    ERR_ABORT_THRESHOLD="$ERR_ABORT_THRESHOLD" \
    RESULTS_ROOT="$SUBDIR" \
    bash scripts/sweep/sweep_all_policies.sh
done

echo ""
echo "============================================================"
echo "  Temporal sweep complete: $ROOT"
echo "============================================================"

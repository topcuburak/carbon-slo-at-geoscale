#!/usr/bin/env bash
# =============================================================================
#  Topology sweep — same hour, different DC selections
# =============================================================================
#
# Runs the full policy × rate sweep across multiple random DC topologies
# (each picked by a different seed) to test whether the carbon savings
# are robust to the choice of datacenters.
#
# Usage:
#   bash scripts/sweep/sweep_topology.sh
#
# Environment overrides:
#   SEEDS="1 7 13 42 99"       set of seeds to sweep
#   SIM_DATETIME="2024-07-15T12:00:00"   wall-clock for CI snapshot
#   ARRIVAL_RATES="30"         arrival rates (one rate keeps it tractable)
#   ERR_ABORT_THRESHOLD=95     pre-flight error threshold (per inner sweep)
#   ROOT=results/topology      where to put per-seed subdirs
# =============================================================================

set -euo pipefail

SEEDS="${SEEDS:-1 7 13 42 99 137 256 999}"
SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
ARRIVAL_RATES="${ARRIVAL_RATES:-30}"
ERR_ABORT_THRESHOLD="${ERR_ABORT_THRESHOLD:-95}"
ROOT="${ROOT:-results/topology_$(date +%Y%m%d_%H%M)}"

mkdir -p "$ROOT"

echo "============================================================"
echo "  Topology sweep"
echo "  Seeds:       $SEEDS"
echo "  Datetime:    $SIM_DATETIME"
echo "  Rates:       $ARRIVAL_RATES"
echo "  Output root: $ROOT"
echo "============================================================"

# Save outer-sweep manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "topology",
  "seeds": [$(echo $SEEDS | sed 's/ /, /g')],
  "sim_datetime": "$SIM_DATETIME",
  "arrival_rates": "$ARRIVAL_RATES",
  "started_at": "$(date -Iseconds)"
}
EOF

for SEED in $SEEDS; do
    SUBDIR="$ROOT/seed_${SEED}"

    echo ""
    echo "============================================================"
    echo "  → Seed $SEED"
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
echo "  Topology sweep complete: $ROOT"
echo "============================================================"

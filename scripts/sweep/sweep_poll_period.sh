#!/usr/bin/env bash
# =============================================================================
#  Poll-period sensitivity sweep — async polling, 4 schedulers × 5 DCs
# =============================================================================
#
# Tests how the provisioner's per-DC poll period affects carbon/SLA
# under async (staggered) polling. Async mode already breaks herd
# symmetry; this sweep isolates the effect of snapshot staleness.
#
# Baseline topology: 4 zones × 5 DCs = 20 DCs total.
#
# Sweep values (POLL_PERIOD_S):
#   0.5  1  2  4  8  16  32
#
# Async stagger (POLL_WAVE_INTERVAL_S) is set to POLL_PERIOD_S / 4 so
# each wave still covers a quarter of the cycle regardless of period.
#
# Usage:
#   bash scripts/sweep/sweep_poll_period.sh
#
# Environment overrides:
#   POLL_PERIODS="0.5 1 2 4 8 16 32"  periods to sweep (doubling)
#   SIM_DATETIME="2024-07-15T12:00"   CI snapshot timestamp
#   ARRIVAL_RATES="200"               arrival rates (space-separated)
#   ROOT=results/poll_period          output root
# =============================================================================

set -euo pipefail

SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
ARRIVAL_RATES="${ARRIVAL_RATES:-200}"
POLL_PERIODS="${POLL_PERIODS:-0.5 1 2 4 8 16 32}"
ROOT="${ROOT:-results/poll_period_$(date +%Y%m%d_%H%M)}"

mkdir -p "$ROOT"

# ── Baseline topology: 4 zones × 5 DCs ───────────────────────────────
PYTHONPATH=proto:. python3 - "$ROOT" << 'PYEOF'
import json, os, sys

out_dir = sys.argv[1]

TOPOLOGY = {
    "europe_north": ["Netherlands", "Belgium", "Lithuania", "Latvia", "Poland"],
    "us_east":      ["US_MISO", "US_DukeProgressE", "US_PJM", "US_DukeCarolinas", "US_NYISO"],
    "us_west":      ["US_PacifiCorpE", "US_Turlock", "US_ElPaso", "US_Avista", "US_APS"],
    "asia":         ["JP_Hokuriku", "PH_Visayas", "JP_Tohoku", "IN_East", "PH_Luzon"],
}

all_dcs = sorted(dc for dcs in TOPOLOGY.values() for dc in dcs)
config = {
    "seed": "poll_period_baseline",
    "n_schedulers": len(TOPOLOGY),
    "n_dcs_total": len(all_dcs),
    "n_nodes_per_dc": 2,
    "topology": TOPOLOGY,
    "all_dcs": all_dcs,
}
path = os.path.join(out_dir, "topology_baseline.json")
with open(path, "w") as f:
    json.dump(config, f, indent=2)
print(f"  baseline: {len(TOPOLOGY)} zones × 5 DCs = {len(all_dcs)} total -> {path}")
PYEOF

TOPO_FILE="$ROOT/topology_baseline.json"

# Save manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "poll_period",
  "poll_mode": "async",
  "poll_periods": "$POLL_PERIODS",
  "topology": "4 zones x 5 DCs = 20 DCs",
  "datetime": "$SIM_DATETIME",
  "arrival_rates": "$ARRIVAL_RATES",
  "started_at": "$(date -Iseconds)"
}
EOF

echo "============================================================"
echo "  Poll-period sensitivity sweep (async)"
echo "  Topology:   4 schedulers × 5 DCs (20 total)"
echo "  Periods:    $POLL_PERIODS s"
echo "  Datetime:   $SIM_DATETIME"
echo "  Rates:      $ARRIVAL_RATES"
echo "  Output:     $ROOT"
echo "============================================================"

for PERIOD in $POLL_PERIODS; do
    # Wave interval: quarter of the full cycle (matches POLL_WAVE_COUNT=4)
    WAVE=$(python3 -c "print($PERIOD / 4)")
    TAG=$(echo "$PERIOD" | tr '.' 'p')  # 0.5 -> 0p5 for filesystem safety
    SUBDIR="$ROOT/period_${TAG}s"

    echo ""
    echo "============================================================"
    echo "  Poll period: ${PERIOD}s   (wave interval: ${WAVE}s)"
    echo "  Output:      $SUBDIR"
    echo "============================================================"

    POLL_MODE="async" \
    POLL_PERIOD_S="$PERIOD" \
    POLL_WAVE_INTERVAL_S="$WAVE" \
    POLL_WAVE_COUNT=4 \
    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    RESULTS_ROOT="$SUBDIR" \
    bash scripts/sweep/sweep_all_policies.sh
done

echo ""
echo "============================================================"
echo "  Poll-period sweep complete: $ROOT"
echo "============================================================"

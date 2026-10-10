#!/usr/bin/env bash
# =============================================================================
#  Scheduler-count sensitivity sweep — 2 vs 4 vs 8 schedulers
# =============================================================================
#
# Tests how the number of independent scheduling regions affects
# the carbon-SLA tradeoff and the async polling advantage. Each
# configuration uses 4 DCs per zone (consistent with spatial
# experiments) but varies the number of zones/schedulers.
#
# Configurations:
#   2_sched  — 2 zones (europe_north, us_east), 8 DCs total
#   4_sched  — 4 zones, 16 DCs total (matches main sweep structure)
#   8_sched  — 8 zones (all regions), 32 DCs total
#
# Usage:
#   bash scripts/sweep/sweep_sched_count.sh
#
# Environment overrides:
#   CONFIGS="2_sched 4_sched 8_sched"   subset to run
#   SIM_DATETIME="2024-07-15T12:00:00"  CI snapshot timestamp
#   ARRIVAL_RATES="200"                 arrival rates per config
#   ROOT=results/sched_count            output root
#   (All sweep_all_policies.sh env vars also apply)
# =============================================================================

set -euo pipefail

SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
ARRIVAL_RATES="${ARRIVAL_RATES:-200}"
ROOT="${ROOT:-results/sched_count_$(date +%Y%m%d_%H%M)}"
CONFIGS="${CONFIGS:-2_sched 4_sched 8_sched}"

mkdir -p "$ROOT"

# ── Generate topology JSONs ──
PYTHONPATH=proto:. python3 - "$ROOT" << 'PYEOF'
import json, sys, os
sys.path.insert(0, ".")

out_dir = sys.argv[1]

TOPOLOGIES = {
    "2_sched": {
        "europe_north": ["Norway", "Netherlands", "Ireland", "Belgium"],
        "us_east":      ["US_PJM", "US_NYISO", "US_ISONE", "US_MISO"],
    },
    "4_sched": {
        "europe_north": ["Netherlands", "Belgium", "Lithuania", "Latvia"],
        "us_east":      ["US_MISO", "US_DukeProgressE", "US_PJM", "US_DukeCarolinas"],
        "us_west":      ["US_PacifiCorpE", "US_Turlock", "US_ElPaso", "US_Avista"],
        "asia":         ["JP_Hokuriku", "PH_Visayas", "JP_Tohoku", "IN_East"],
    },
    "8_sched": {
        "europe_north":  ["Norway", "Netherlands", "Ireland", "Belgium"],
        "europe_south":  ["France", "Germany", "Spain", "Switzerland"],
        "us_east":       ["US_PJM", "US_NYISO", "US_ISONE", "US_MISO"],
        "us_south":      ["US_ERCOT", "US_TVA", "US_Southern", "US_SPP"],
        "us_west":       ["US_CAISO", "US_BPA", "US_APS", "US_Avista"],
        "middle_east":   ["Turkey", "Israel", "Greece", "Romania"],
        "asia":          ["JP_Tokyo", "SouthKorea", "AU_NSW", "Singapore"],
        "south_america": ["BR_South", "Chile", "Peru", "BR_Central"],
    },
}

for name, topo in TOPOLOGIES.items():
    all_dcs = sorted(dc for dcs in topo.values() for dc in dcs)
    config = {
        "seed": f"sched_count_{name}",
        "n_schedulers": len(topo),
        "n_dcs_total": len(all_dcs),
        "n_nodes_per_dc": 2,
        "topology": topo,
        "all_dcs": all_dcs,
    }
    path = os.path.join(out_dir, f"topology_{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  {name}: {len(topo)} zones x 4 DCs = {len(all_dcs)} total -> {path}")
PYEOF

# Save manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "sched_count",
  "configs": [$(echo $CONFIGS | sed 's/ /", "/g; s/^/"/; s/$/"/')],
  "datetime": "$SIM_DATETIME",
  "arrival_rates": "$ARRIVAL_RATES",
  "started_at": "$(date -Iseconds)"
}
EOF

echo "============================================================"
echo "  Scheduler-count sensitivity sweep"
echo "  Configs:     $CONFIGS"
echo "  Datetime:    $SIM_DATETIME"
echo "  Rates:       $ARRIVAL_RATES"
echo "  Output root: $ROOT"
echo "============================================================"

for CFG in $CONFIGS; do
    TOPO_FILE="$ROOT/topology_${CFG}.json"
    SUBDIR="$ROOT/${CFG}"

    if [ ! -f "$TOPO_FILE" ]; then
        echo "ERROR: topology file not found: $TOPO_FILE"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Config: $CFG"
    echo "  Topology: $TOPO_FILE"
    echo "  Output:   $SUBDIR"
    echo "============================================================"

    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    RESULTS_ROOT="$SUBDIR" \
    bash scripts/sweep/sweep_all_policies.sh
done

echo ""
echo "============================================================"
echo "  Scheduler-count sweep complete: $ROOT"
echo "============================================================"

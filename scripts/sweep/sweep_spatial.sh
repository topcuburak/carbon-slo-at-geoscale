#!/usr/bin/env bash
# =============================================================================
#  Spatial variation sweep — 4 hardcoded DC topologies
# =============================================================================
#
# Tests carbon-aware scheduling robustness across geographically diverse
# DC placements. Each experiment uses 4 scheduler zones × 4 DCs = 16 DCs,
# but the zones and DCs are chosen to vary cross-zone RTT and CI diversity.
#
# Experiments:
#   global_spread  — one zone per continent (avg RTT 175 ms, max 280 ms)
#   us_europe      — tight transatlantic cluster (avg 87 ms, max 140 ms)
#   asia_pacific   — east-to-west corridor (avg 126 ms, max 190 ms)
#   high_latency   — maximally dispersed (avg 178 ms, max 280 ms)
#
# Usage:
#   bash scripts/sweep/sweep_spatial.sh
#
# Environment overrides:
#   EXPERIMENTS="global_spread us_europe"   subset of experiments to run
#   SIM_DATETIME="2024-07-15T12:00:00"     CI snapshot timestamp
#   ARRIVAL_RATES="200"                     arrival rates per experiment
#   ROOT=results/spatial                    where to put per-experiment subdirs
#   (All sweep_all_policies.sh env vars also apply: POLL_MODE, BS_CAP, etc.)
# =============================================================================

set -euo pipefail

SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
ARRIVAL_RATES="${ARRIVAL_RATES:-200}"
ROOT="${ROOT:-results/spatial_$(date +%Y%m%d_%H%M)}"

# Which experiments to run (space-separated)
EXPERIMENTS="${EXPERIMENTS:-global_spread us_europe asia_pacific high_latency}"

mkdir -p "$ROOT"

# ── Generate topology JSONs via inline Python ──
PYTHONPATH=proto:. python3 - "$ROOT" << 'PYEOF'
import json, sys, os
sys.path.insert(0, ".")

out_dir = sys.argv[1]

TOPOLOGIES = {
    "global_spread": {
        "europe_north":  ["Norway", "Netherlands", "Ireland", "SE_North"],
        "us_east":       ["US_PJM", "US_NYISO", "US_ISONE", "US_MISO"],
        "asia":          ["JP_Tokyo", "SouthKorea", "AU_NSW", "Singapore"],
        "south_america": ["BR_South", "Chile", "Peru", "BR_Central"],
    },
    "us_europe": {
        "us_east":       ["US_PJM", "US_NYISO", "US_MISO", "US_DukeCarolinas"],
        "us_west":       ["US_CAISO", "US_BPA", "US_APS", "US_Avista"],
        "europe_north":  ["Norway", "Netherlands", "Belgium", "Ireland"],
        "europe_south":  ["France", "Germany", "Spain", "Switzerland"],
    },
    "asia_pacific": {
        "asia":          ["JP_Tokyo", "IN_South", "AU_NSW", "SouthKorea"],
        "us_west":       ["US_CAISO", "US_BPA", "US_PacifiCorpE", "US_Avista"],
        "middle_east":   ["Turkey", "Israel", "Greece", "Romania"],
        "europe_south":  ["France", "Germany", "IT_North", "Spain"],
    },
    "high_latency": {
        "asia":          ["JP_Tokyo", "Singapore", "AU_NSW", "IN_North"],
        "south_america": ["BR_South", "Chile", "Peru", "Nicaragua"],
        "europe_north":  ["Norway", "Finland", "Ireland", "Lithuania"],
        "middle_east":   ["Turkey", "SouthAfrica", "Israel", "Cyprus"],
    },
}

for name, topo in TOPOLOGIES.items():
    all_dcs = sorted(dc for dcs in topo.values() for dc in dcs)
    config = {
        "seed": f"spatial_{name}",
        "n_schedulers": len(topo),
        "n_dcs_total": len(all_dcs),
        "n_nodes_per_dc": 2,
        "topology": topo,
        "all_dcs": all_dcs,
    }
    path = os.path.join(out_dir, f"topology_{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {path}  ({len(topo)} zones × {len(all_dcs)//len(topo)} DCs = {len(all_dcs)} total)")
PYEOF

# Save manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "spatial",
  "datetime": "$SIM_DATETIME",
  "arrival_rates": "$ARRIVAL_RATES",
  "experiments": [$(echo $EXPERIMENTS | sed 's/ /", "/g; s/^/"/; s/$/"/')],
  "started_at": "$(date -Iseconds)"
}
EOF

echo "============================================================"
echo "  Spatial variation sweep"
echo "  Experiments:   $EXPERIMENTS"
echo "  Datetime:      $SIM_DATETIME"
echo "  Rates:         $ARRIVAL_RATES"
echo "  Output root:   $ROOT"
echo "============================================================"

for EXP in $EXPERIMENTS; do
    TOPO_FILE="$ROOT/topology_${EXP}.json"
    SUBDIR="$ROOT/${EXP}"

    if [ ! -f "$TOPO_FILE" ]; then
        echo "ERROR: topology file not found: $TOPO_FILE"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Experiment: $EXP"
    echo "  Topology:   $TOPO_FILE"
    echo "  Output:     $SUBDIR"
    echo "============================================================"

    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    RESULTS_ROOT="$SUBDIR" \
    bash scripts/sweep/sweep_all_policies.sh
done

echo ""
echo "============================================================"
echo "  Spatial sweep complete: $ROOT"
echo "============================================================"

#!/usr/bin/env bash
# =============================================================================
#  Spatial sensitivity sweep — parallel orchestrator
# =============================================================================
#
# Runs sweep_parallel.sh (7 policies in parallel within rate) on each of N
# hardcoded geographic topologies. Topologies vary in cross-zone RTT and
# carbon-intensity diversity to study how policy ranking depends on the
# physical placement of DCs.
#
# Topologies run sequentially; policies run in parallel within each
# topology. Estimated wallclock: ~30 min × 4 topologies = ~2 hours.
#
# Usage:
#   bash scripts/sweep/sweep_spatial_parallel.sh
#
# Environment overrides:
#   EXPERIMENTS="global_spread us_europe asia_pacific high_latency"
#   SIM_DATETIME="2024-07-15T12:00:00"
#   ARRIVAL_RATES="200"
#   N_NODES_PER_DC=10
#   TTFT_BUDGET=2000 TPOT_BUDGET=50 BS_CAP=128
#   ROOT=results/spatial_parallel_<timestamp>
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

SIM_DATETIME="${SIM_DATETIME:-2024-07-15T12:00:00}"
ARRIVAL_RATES="${ARRIVAL_RATES:-200}"
ROOT="${ROOT:-results/spatial_parallel_$(date +%Y%m%d_%H%M)}"
EXPERIMENTS="${EXPERIMENTS:-global_spread us_europe asia_pacific high_latency}"

export N_NODES_PER_DC="${N_NODES_PER_DC:-10}"
export TTFT_BUDGET="${TTFT_BUDGET:-2000}"
export TPOT_BUDGET="${TPOT_BUDGET:-50}"
export BS_CAP="${BS_CAP:-128}"

mkdir -p "$ROOT"

# ── Generate per-experiment topology JSONs ──
# Each topology is 4 zones × 8 DCs = 32 DCs, matching the new main sweep.
PYTHONPATH=proto:. python3 - "$ROOT" "$N_NODES_PER_DC" << 'PYEOF'
import json, os, sys
out_dir = sys.argv[1]
n_nodes_per_dc = int(sys.argv[2])

# All DCs below MUST be in common.network.SCHEDULER_GROUPS for the simulator.
# Aim is 8 DCs per zone × 4 zones = 32 DCs total. south_america and
# middle_east only have 7 DCs in SCHEDULER_GROUPS, so those zones use 7.
# Topologies that include those zones end up with 31 DCs total.
TOPOLOGIES = {
    "global_spread": {
        "europe_north": ["Norway","Netherlands","Ireland","SE_North","DK_East","DK_West","Belgium","Lithuania"],
        "us_east":      ["US_PJM","US_NYISO","US_ISONE","US_MISO","US_DukeCarolinas","US_DukeProgressE","US_DukeProgressW","US_SCEG"],
        "asia":         ["JP_Hokkaido","JP_Tohoku","JP_Hokuriku","JP_Kansai","Singapore","IN_East","PH_Luzon","PH_Visayas"],
        "south_america":["BR_South","BR_Central","BR_North","BR_NorthEast","Chile","Peru","Nicaragua"],  # 7 DCs (max available)
    },
    "us_europe": {
        "us_east":      ["US_PJM","US_NYISO","US_MISO","US_DukeCarolinas","US_ISONE","US_DukeProgressE","US_DukeProgressW","US_SCEG"],
        "us_west":      ["US_CAISO","US_BPA","US_APS","US_Avista","US_LADWP","US_Turlock","US_PacifiCorpE","US_ElPaso"],
        "europe_north": ["Norway","Netherlands","Belgium","Ireland","DK_East","DK_West","SE_North","Lithuania"],
        "europe_south": ["France","Germany","Spain","Switzerland","IT_North","Austria","Czechia","Hungary"],
    },
    "asia_pacific": {
        "asia":         ["JP_Hokkaido","JP_Tohoku","JP_Hokuriku","JP_Kansai","Singapore","IN_East","PH_Luzon","PH_Visayas"],
        "us_west":      ["US_CAISO","US_BPA","US_PacifiCorpE","US_Avista","US_LADWP","US_Turlock","US_APS","US_ElPaso"],
        "middle_east":  ["Turkey","Israel","Greece","Romania","Bulgaria","Cyprus","SouthAfrica"],  # 7 DCs (max available)
        "europe_south": ["France","Germany","IT_North","Spain","Austria","Czechia","Hungary","Switzerland"],
    },
    "high_latency": {
        "asia":         ["JP_Hokkaido","Singapore","IN_East","PH_Luzon","PH_Visayas","JP_Hokuriku","JP_Tohoku","JP_Kansai"],
        "south_america":["BR_South","Chile","Peru","BR_Central","BR_North","BR_NorthEast","Nicaragua"],  # 7 DCs
        "europe_north": ["Norway","Finland","Ireland","Lithuania","DK_East","DK_West","Belgium","Estonia"],
        "middle_east":  ["Turkey","Israel","Cyprus","SouthAfrica","Greece","Romania","Bulgaria"],       # 7 DCs
    },
}

# Validate against SCHEDULER_GROUPS to fail fast on typos.
sys.path.insert(0, ".")
from common.network import SCHEDULER_GROUPS
all_known = {dc for dcs in SCHEDULER_GROUPS.values() for dc in dcs}
for name, topo in TOPOLOGIES.items():
    bad = [dc for dcs in topo.values() for dc in dcs if dc not in all_known]
    if bad:
        print(f"  WARNING {name}: unknown DCs (will skip in topology): {bad}", file=sys.stderr)
        # Filter out unknowns
        for zone in topo:
            topo[zone] = [d for d in topo[zone] if d in all_known]

for name, topo in TOPOLOGIES.items():
    all_dcs = sorted(dc for dcs in topo.values() for dc in dcs)
    config = {
        "seed": f"spatial_parallel_{name}",
        "n_schedulers": len(topo),
        "n_dcs_total": len(all_dcs),
        "n_nodes_per_dc": n_nodes_per_dc,
        "topology": topo,
        "all_dcs": all_dcs,
    }
    path = os.path.join(out_dir, f"topology_{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    counts_per_zone = {z: len(d) for z, d in topo.items()}
    print(f"  {name}: zones={counts_per_zone} total={len(all_dcs)} → {path}")
PYEOF

# Save manifest
cat > "$ROOT/manifest.json" <<EOF
{
  "type": "spatial_parallel",
  "datetime": "$SIM_DATETIME",
  "arrival_rates": "$ARRIVAL_RATES",
  "experiments": [$(echo $EXPERIMENTS | sed 's/ /", "/g; s/^/"/; s/$/"/')],
  "n_nodes_per_dc": $N_NODES_PER_DC,
  "ttft_budget_ms": $TTFT_BUDGET,
  "tpot_budget_ms": $TPOT_BUDGET,
  "bs_cap": $BS_CAP,
  "started_at": "$(date -Iseconds)"
}
EOF

echo "============================================================"
echo "  Spatial parallel sweep"
echo "  Experiments:   $EXPERIMENTS"
echo "  Datetime:      $SIM_DATETIME"
echo "  Rates:         $ARRIVAL_RATES"
echo "  Nodes/DC:      $N_NODES_PER_DC"
echo "  TTFT budget:   ${TTFT_BUDGET} ms"
echo "  TPOT budget:   ${TPOT_BUDGET} ms"
echo "  BS_CAP:        $BS_CAP"
echo "  Output root:   $ROOT"
echo "============================================================"

EXP_IDX=0
for EXP in $EXPERIMENTS; do
    TOPO_FILE="$ROOT/topology_${EXP}.json"
    SUBDIR="$ROOT/${EXP}"
    LOG="$ROOT/${EXP}.log"

    if [ ! -f "$TOPO_FILE" ]; then
        echo "ERROR: topology file not found: $TOPO_FILE"
        continue
    fi

    # All experiments reuse POLICY_SLOT_BASE=0 (slot range 0-6, ports
    # 50100-56300). Experiments run sequentially so concurrency isolation
    # is unnecessary; a 60s pause lets any TIME_WAIT'd sockets clear.
    # The earlier EXP_IDX*8 scheme overflowed the 16-bit TCP port limit
    # at experiment ≥ 2 (port = 50100 + slot*1000 > 65535).
    export POLICY_SLOT_BASE=0
    if [ "$EXP_IDX" -gt 0 ]; then
        echo "  (waiting 60s for TIME_WAIT to clear before next experiment…)"
        sleep 60
    fi
    EXP_IDX=$((EXP_IDX + 1))

    echo
    echo "============================================================"
    echo "  Experiment: $EXP"
    echo "  Topology:   $TOPO_FILE"
    echo "  Slot base:  $POLICY_SLOT_BASE  (ports $((50100 + POLICY_SLOT_BASE * 1000))+)"
    echo "  Output:     $SUBDIR"
    echo "============================================================"
    t0=$(date +%s)

    SIM_DATETIME="$SIM_DATETIME" \
    ARRIVAL_RATES="$ARRIVAL_RATES" \
    TOPOLOGY_JSON="$TOPO_FILE" \
    ROOT="$SUBDIR" \
    POLICY_SLOT_BASE="$POLICY_SLOT_BASE" \
    bash scripts/sweep/sweep_parallel.sh > "$LOG" 2>&1

    dt=$(( $(date +%s) - t0 ))
    echo "  $EXP done in ${dt}s"
done

echo
echo "============================================================"
echo "  Spatial parallel sweep complete: $ROOT"
echo "============================================================"

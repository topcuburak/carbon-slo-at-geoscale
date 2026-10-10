"""Generate a randomized experiment configuration.

Randomly selects 4 scheduler regions and 5 DCs per region from the
full 104-DC topology. Outputs a JSON config for sweep scripts.

Usage:
    python scripts/sweep/generate_experiment_config.py --seed 42
    python scripts/sweep/generate_experiment_config.py --seed 123 --n-schedulers 4 --n-dcs-per 5
"""

import argparse
import json
import random
import sys

sys.path.insert(0, ".")
from common.network import SCHEDULER_GROUPS


# Fixed regions — geographically diverse with varying carbon intensity
DEFAULT_REGIONS = [
    "europe_north",  # clean — hydro/wind
    "us_east",       # moderate — mixed grid
    "us_west",       # variable — solar + gas
    "asia",          # dirty — coal-heavy in some regions
]


def generate_config(regions: list = None, n_dcs_per: int = 5,
                    n_nodes_per_dc: int = 2, seed: int = 42) -> dict:
    rng = random.Random(seed)

    selected_regions = regions or DEFAULT_REGIONS

    # Pick n_dcs_per DCs from each selected region
    topology = {}
    all_selected_dcs = []
    for region in selected_regions:
        available_dcs = list(SCHEDULER_GROUPS[region])
        rng.shuffle(available_dcs)
        picked = available_dcs[:min(n_dcs_per, len(available_dcs))]
        topology[region] = picked
        all_selected_dcs.extend(picked)

    config = {
        "seed": seed,
        "n_schedulers": len(selected_regions),
        "n_dcs_total": len(all_selected_dcs),
        "n_nodes_per_dc": n_nodes_per_dc,
        "topology": topology,
        "all_dcs": sorted(all_selected_dcs),
    }
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate experiment config")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Fixed regions (default: europe_north us_east us_west asia)")
    parser.add_argument("--n-dcs-per", type=int, default=5)
    parser.add_argument("--n-nodes-per-dc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: print to stdout)")
    args = parser.parse_args()

    config = generate_config(
        regions=args.regions,
        n_dcs_per=args.n_dcs_per,
        n_nodes_per_dc=args.n_nodes_per_dc,
        seed=args.seed,
    )

    print(f"Experiment config (seed={args.seed}):", file=sys.stderr)
    print(f"  Schedulers: {config['n_schedulers']}", file=sys.stderr)
    print(f"  DCs total:  {config['n_dcs_total']}", file=sys.stderr)
    for region, dcs in config["topology"].items():
        print(f"  {region}: {dcs}", file=sys.stderr)

    output = json.dumps(config, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

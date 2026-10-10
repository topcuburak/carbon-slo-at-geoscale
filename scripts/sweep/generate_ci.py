#!/usr/bin/env python3
"""Generate carbon intensity values for all DCs at a specific or random hour.

Outputs shell-compatible CI[] assignments that can be eval'd by bash scripts.

Usage:
    # Specific hour (UTC):
    python scripts/sweep/generate_ci.py "2024-10-15T06"

    # Random hour from a given year:
    python scripts/sweep/generate_ci.py random

    # Random hour from a specific year:
    SIM_YEAR=2023 python scripts/sweep/generate_ci.py random

    # Custom carbon data path:
    CARBON_DATA_ROOT=carbon_intensity python scripts/sweep/generate_ci.py "2024-10-15T06"

Output (to stdout, eval-friendly):
    SIM_DATETIME="2024-10-15T06:00:00"
    CI[DK_East]=33.4
    CI[France]=6.8
    ...
"""

import os
import random
import sys
from datetime import datetime

# Add project root to path so we can import common.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common.carbon import CarbonLoader, REGION_CSV_MAP
from common.network import SCHEDULER_GROUPS


def pick_random_hour(year: int, seed: int | None = None) -> datetime:
    """Pick a random valid hour from the given year."""
    rng = random.Random(seed)
    month = rng.randint(1, 12)
    # Days per month (handle leap years)
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = rng.randint(1, max_day)
    hour = rng.randint(0, 23)
    return datetime(year, month, day, hour, 0, 0)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "random"
    carbon_root = os.environ.get("CARBON_DATA_ROOT", "carbon_intensity")
    default_year = int(os.environ.get("SIM_YEAR", "2024"))
    seed = os.environ.get("SIM_SEED")
    seed = int(seed) if seed else None

    # Determine datetime
    if arg.lower() == "random":
        dt = pick_random_hour(default_year, seed=seed)
    else:
        # Parse "YYYY-MM-DDTHH" or "YYYY-MM-DDTHH:MM" or "YYYY-MM-DDTHH:MM:SS"
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
            try:
                dt = datetime.strptime(arg, fmt)
                break
            except ValueError:
                continue
        else:
            print(f"ERROR: Cannot parse datetime: {arg}", file=sys.stderr)
            print(f"Expected format: YYYY-MM-DDTHH (e.g., 2024-10-15T06)", file=sys.stderr)
            sys.exit(1)

    year = dt.year

    # Determine which regions have data for this year
    all_dcs = [dc for group in SCHEDULER_GROUPS.values() for dc in group]
    available = []
    missing = []
    for dc in all_dcs:
        if dc not in REGION_CSV_MAP:
            missing.append(dc)
            continue
        subdir, prefix = REGION_CSV_MAP[dc]
        fpath = os.path.join(carbon_root, subdir, f"{prefix}_{year}_hourly.csv")
        if os.path.exists(fpath):
            available.append(dc)
        else:
            missing.append(dc)

    if missing:
        print(f"# WARNING: {len(missing)} DCs missing {year} data: {missing[:5]}...",
              file=sys.stderr)

    if not available:
        print(f"ERROR: No carbon data found for year {year} in {carbon_root}",
              file=sys.stderr)
        sys.exit(1)

    # Load carbon data
    loader = CarbonLoader(carbon_root, year=year, regions=available)

    # Output shell-compatible assignments
    print(f'SIM_DATETIME="{dt.strftime("%Y-%m-%dT%H:%M:%S")}"')
    print(f'SIM_YEAR="{year}"')

    for group_name in sorted(SCHEDULER_GROUPS.keys()):
        print(f"# {group_name}")
        for dc in SCHEDULER_GROUPS[group_name]:
            if dc in available:
                ci = loader.get_ci(dc, dt)
                print(f"CI[{dc}]={ci:.1f}")
            else:
                # Fallback: use 0 and warn
                print(f"CI[{dc}]=0.0  # NO DATA")

    # Summary to stderr
    cis = [loader.get_ci(dc, dt) for dc in available]
    print(f"\n# {dt.strftime('%Y-%m-%d %H:%M')} UTC  |  "
          f"{len(available)} DCs  |  "
          f"CI range: {min(cis):.1f} - {max(cis):.1f}  |  "
          f"avg: {sum(cis)/len(cis):.1f} gCO2/kWh",
          file=sys.stderr)


if __name__ == "__main__":
    main()

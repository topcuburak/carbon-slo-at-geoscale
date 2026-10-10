#!/usr/bin/env python3
"""The carbon archive: the number every carbon result is multiplied by.

Energy is measured against a fitted model and validated on a testbed. Carbon
intensity is not measured at all -- it is read out of this archive, and if a
site reads the wrong row, or quietly falls back to a fleet default, the whole
result moves with it and nothing looks wrong.

The most valuable check here is the boring one: does the archive actually
cover all 102 datacentres in the topology. A site the map does not name takes
`--default-ci 400` at launch, and 400 is a plausible-looking number.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_carbon.py
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

from common.carbon import REGION_CSV_MAP, CarbonLoader              # noqa: E402
from common.network import SCHEDULER_GROUPS                         # noqa: E402

ARCHIVE = ROOT / "carbon_intensity"
YEAR = 2024
CI_COL = "Carbon Intensity gCO₂eq/kWh (direct)"

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except BaseException as e:                                  # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


def all_dcs():
    return [d for g in SCHEDULER_GROUPS for d in SCHEDULER_GROUPS[g]]


def fake_archive(rows, subdir="fakeland", prefix="FK"):
    """A one-region archive on disk, in the format the loader expects."""
    root = Path(tempfile.mkdtemp())
    d = root / subdir
    d.mkdir(parents=True)
    with open(d / f"{prefix}_{YEAR}_hourly.csv", "w") as f:
        f.write(f"Datetime (UTC),{CI_COL}\n")
        for dt, ci in rows:
            f.write(f"{dt},{ci}\n")
    return str(root)


# ───────────────────────── the map against the topology ─────────────────────

def test_the_map_names_every_datacentre_in_the_topology():
    missing = [d for d in all_dcs() if d not in REGION_CSV_MAP]
    assert not missing, \
        (f"{len(missing)} site(s) have no carbon mapping and would silently "
         f"take --default-ci: {missing[:8]}")


def test_the_map_names_nothing_that_is_not_in_the_topology():
    # A stale entry is not harmful, but it means the map and the topology have
    # drifted, and the next site added may be added to only one of them.
    extra = sorted(set(REGION_CSV_MAP) - set(all_dcs()))
    assert not extra, f"the map names sites the topology does not: {extra}"


def test_the_topology_is_the_size_the_paper_reports():
    # 102, not 104: Hong Kong and Indonesia were retired on 2026-09-02 for
    # carrying a constant instead of a measurement. See
    # carbon_intensity/_retired/README.md, and update the paper's zone count.
    dcs = all_dcs()
    assert len(dcs) == 102, len(dcs)
    assert len(set(dcs)) == 102, "a datacentre appears in two groups"


def test_every_datacentre_has_a_file_in_the_archive():
    missing = []
    for dc in all_dcs():
        subdir, prefix = REGION_CSV_MAP[dc]
        if not (ARCHIVE / subdir / f"{prefix}_{YEAR}_hourly.csv").is_file():
            missing.append(dc)
    assert not missing, \
        (f"{len(missing)} site(s) are mapped but have no {YEAR} file, so they "
         f"fall back to the fleet default: {missing[:8]}")


# ──────────────────────────── the real archive ──────────────────────────────

def test_the_whole_fleet_loads_and_every_site_reports_a_real_number():
    loader = CarbonLoader(str(ARCHIVE), year=YEAR, regions=all_dcs())
    assert len(loader.get_all_regions()) == 102, len(loader.get_all_regions())
    when = datetime(2024, 7, 15, 12, 0, 0)          # the sweep's timestamp
    for dc in all_dcs():
        ci = loader.get_ci(dc, when)
        assert 0.0 < ci < 2000.0, f"{dc} reports {ci} gCO2/kWh"


def test_the_fleet_spans_a_range_worth_scheduling_over():
    # If every site read the same number there would be nothing for a
    # carbon-aware policy to exploit, and a flat archive is exactly what a
    # silent fallback produces.
    loader = CarbonLoader(str(ARCHIVE), year=YEAR, regions=all_dcs())
    when = datetime(2024, 7, 15, 12, 0, 0)
    vals = [loader.get_ci(d, when) for d in all_dcs()]
    assert max(vals) / min(vals) > 5.0, \
        f"the fleet spans only {min(vals):.0f}-{max(vals):.0f} gCO2/kWh"
    assert len({round(v, 3) for v in vals}) > 50, \
        "many sites report identical intensity; the archive may not be read"


def test_the_same_site_moves_over_the_day():
    # Temporal variation is a paper axis: the same experiment at another
    # timestamp is how it is studied.
    loader = CarbonLoader(str(ARCHIVE), year=YEAR, regions=["France", "Germany"])
    day = [loader.get_ci("Germany", datetime(2024, 7, 15, h, 0, 0))
           for h in range(24)]
    assert len(set(day)) > 1, "Germany reports one value for the whole day"


def test_loading_a_subset_loads_only_that_subset()  :
    loader = CarbonLoader(str(ARCHIVE), year=YEAR, regions=["France"])
    assert loader.get_all_regions() == ["France"]
    try:
        loader.get_ci("Germany", datetime(2024, 7, 15, 12, 0, 0))
    except ValueError:
        return
    raise AssertionError("a region that was never loaded answered")


def test_an_unknown_region_is_refused_at_construction():
    try:
        CarbonLoader(str(ARCHIVE), year=YEAR, regions=["Atlantis"])
    except ValueError as e:
        assert "Atlantis" in str(e), e
        return
    raise AssertionError("an unknown region was accepted")


def test_summary_describes_what_was_loaded():
    loader = CarbonLoader(str(ARCHIVE), year=YEAR, regions=["France", "Poland"])
    s = loader.summary()
    assert "France" in s and "Poland" in s, s
    assert str(YEAR) in s, s


# ───────────────────────────── lookup and fallback ──────────────────────────

def test_an_exact_hour_returns_that_hour_and_not_an_average():
    root = fake_archive([
        ("2024-03-04 00:00:00", 100.0),
        ("2024-03-04 01:00:00", 200.0),
        ("2024-03-04 02:00:00", 300.0),
    ])
    REGION_CSV_MAP["__test__"] = ("fakeland", "FK")
    try:
        loader = CarbonLoader(root, year=YEAR, regions=["__test__"])
        assert loader.get_ci("__test__", datetime(2024, 3, 4, 1, 0, 0)) == 200.0
        # and the minute is ignored: the archive is hourly
        assert loader.get_ci("__test__", datetime(2024, 3, 4, 1, 45, 0)) == 200.0
    finally:
        del REGION_CSV_MAP["__test__"]


def test_a_missing_hour_falls_back_to_that_day():
    root = fake_archive([
        ("2024-03-04 00:00:00", 100.0),
        ("2024-03-04 01:00:00", 300.0),
        ("2024-03-05 00:00:00", 999.0),
    ])
    REGION_CSV_MAP["__test__"] = ("fakeland", "FK")
    try:
        loader = CarbonLoader(root, year=YEAR, regions=["__test__"])
        got = loader.get_ci("__test__", datetime(2024, 3, 4, 7, 0, 0))
        assert got == 200.0, f"expected the 4th's mean (200), got {got}"
    finally:
        del REGION_CSV_MAP["__test__"]


def test_a_missing_day_falls_back_to_that_month():
    root = fake_archive([
        ("2024-03-04 00:00:00", 100.0),
        ("2024-03-05 00:00:00", 300.0),
        ("2024-04-01 00:00:00", 999.0),
    ])
    REGION_CSV_MAP["__test__"] = ("fakeland", "FK")
    try:
        loader = CarbonLoader(root, year=YEAR, regions=["__test__"])
        got = loader.get_ci("__test__", datetime(2024, 3, 20, 7, 0, 0))
        assert got == 200.0, f"expected March's mean (200), got {got}"
    finally:
        del REGION_CSV_MAP["__test__"]


def test_no_data_for_the_month_raises_rather_than_inventing_a_number():
    # The last resort must be an error. Returning a fleet-wide average here
    # would put a fabricated intensity under a carbon result.
    root = fake_archive([("2024-03-04 00:00:00", 100.0)])
    REGION_CSV_MAP["__test__"] = ("fakeland", "FK")
    try:
        loader = CarbonLoader(root, year=YEAR, regions=["__test__"])
        try:
            loader.get_ci("__test__", datetime(2024, 9, 1, 0, 0, 0))
        except ValueError:
            return
        raise AssertionError("a month with no data returned a number")
    finally:
        del REGION_CSV_MAP["__test__"]


def test_a_blank_reading_is_skipped_not_read_as_zero():
    # Zero would be the greenest site in the fleet, and a carbon-aware policy
    # would send it everything.
    root = fake_archive([
        ("2024-03-04 00:00:00", ""),
        ("2024-03-04 01:00:00", 300.0),
    ])
    REGION_CSV_MAP["__test__"] = ("fakeland", "FK")
    try:
        loader = CarbonLoader(root, year=YEAR, regions=["__test__"])
        # the blank hour is absent, so it falls back to the day -- 300, not 150
        assert loader.get_ci("__test__", datetime(2024, 3, 4, 0, 0, 0)) == 300.0
        assert loader.get_ci("__test__", datetime(2024, 3, 4, 1, 0, 0)) == 300.0
    finally:
        del REGION_CSV_MAP["__test__"]


def test_a_missing_file_is_an_error_and_not_an_empty_region():
    REGION_CSV_MAP["__test__"] = ("nowhere", "ZZ")
    try:
        CarbonLoader(str(ARCHIVE), year=YEAR, regions=["__test__"])
    except (FileNotFoundError, OSError):
        return
    finally:
        del REGION_CSV_MAP["__test__"]
    raise AssertionError("a missing archive file loaded as an empty region")


# ───────────────────────── placeholders, retired ────────────────────────────

RETIRED = ("HongKong", "Indonesia")


def test_the_retired_zones_are_out_of_every_map():
    # They read a constant, so a scheduler could neither prefer nor avoid them
    # on carbon. Out of the topology, out of the carbon map, out of the PUE
    # table -- all three, or one of them puts the zone back.
    import json
    pue = json.loads((ROOT / "common" / "pue_by_dc.json").read_text())
    for z in RETIRED:
        assert z not in all_dcs(), f"{z} is back in the topology"
        assert z not in REGION_CSV_MAP, f"{z} is back in the carbon map"
        assert z not in pue, f"{z} is back in the PUE table"


def test_the_retired_data_is_kept_and_explained():
    # Moved, not deleted: the decision has to be reversible and the reason has
    # to survive the person who made it.
    d = ROOT / "carbon_intensity" / "_retired"
    assert (d / "README.md").is_file(), "no note explaining what was retired"
    for sub in ("hong_kong", "indonesia"):
        assert (d / sub).is_dir(), f"{sub} was deleted rather than retired"


def test_no_datacentre_in_use_reads_a_constant():
    # The check that keeps the archive honest as zones are added. A file that
    # never moves is a placeholder, and a datacentre reading one is invisible
    # to every carbon decision while still counting in every average.
    import csv
    worst = []
    for dc in all_dcs():
        subdir, prefix = REGION_CSV_MAP[dc]
        f = ARCHIVE / subdir / f"{prefix}_{YEAR}_hourly.csv"
        vals = []
        with open(f) as fh:
            for row in csv.DictReader(fh):
                v = (row.get(CI_COL) or "").strip()
                if v:
                    vals.append(v)
        if not vals:
            worst.append((dc, 1.0, "no readings"))
            continue
        longest = run = 1
        for a, b in zip(vals, vals[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        share = longest / len(vals)
        if share > 0.25:
            worst.append((dc, share, f"{longest}h of {len(vals)} constant"))
    assert not worst, \
        ("datacentre(s) whose intensity barely moves: "
         + "; ".join(f"{d} ({m}, {s*100:.0f}% of the year)" for d, s, m in worst))


# ─────────────────── what the launcher does with all this ───────────────────

def test_the_launcher_reads_the_archive_rather_than_defaulting():
    # datacenter/launch.py:load_ci swallows both a failed import and an
    # unreadable archive and returns the fallback for every site -- a run then
    # looks perfectly healthy with a flat fleet. This is the check that it did
    # not happen.
    from datacenter.launch import load_ci
    dcs = list(SCHEDULER_GROUPS["us_east"])
    ci, missing = load_ci(dcs, "2024-07-15T12:00:00", fallback=400.0)
    assert not missing, f"sites with no data: {missing}"
    assert len(ci) == len(dcs), (len(ci), len(dcs))
    assert len({round(v, 3) for v in ci.values()}) > 1, \
        f"every us_east site read the same intensity: {set(ci.values())}"
    assert not all(v == 400.0 for v in ci.values()), \
        "every site took the fallback; the archive was not read"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        check(t.__name__)(t)
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    sys.exit(1 if FAIL else 0)

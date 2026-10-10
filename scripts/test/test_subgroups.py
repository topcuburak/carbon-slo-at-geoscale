#!/usr/bin/env python3
"""Several schedulers in one region: the "group_parents" split of a topology.

The 8-scheduler topology study has to change the scheduler count and nothing
else, so a region's DCs are divided between two schedulers instead of adding
new regions with new DCs. These check that the split does exactly that:

  ownership   each sub-group owns its listed DCs and nothing else
  geography   two halves of one region are the region's own round trip apart,
              and each half is as far from every other region as the whole
  the view    the provisioner's RTT matrix and the estimator's lookup agree
  guard rails a DC outside the parent region, or a parent that also runs as a
              scheduler, is refused rather than priced at the wrong RTT
  no split    a topology without "group_parents" changes nothing

common/network.py applies the split at import, so every case runs in a fresh
interpreter with SIM_TOPOLOGY_FILE set.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_subgroups.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAME16 = ROOT / "scripts/sweep/topologies/v9_exp3_8sched_same16.json"
BASE = ROOT / "scripts/sweep/topologies/v9_real_pue.json"

PROBE = """
import json
import common.network as n
from scheduler.policies.common.estimators import _lookup_rtt_ms
m = n.get_all_base_rtts()
print(json.dumps(dict(
    group={dc: n.REGION_TO_GROUP[dc] for dc in ("Netherlands", "DK_East", "BR_North", "Chile")},
    owns={g: n.SCHEDULER_GROUPS.get(g) for g in ("europe_north_a", "europe_north_b")},
    rtt_halves=n.get_rtt_ms("europe_north_a", "europe_north_b"),
    rtt_sa_halves=n.get_rtt_ms("south_america_a", "south_america_b"),
    rtt_cross=n.get_rtt_ms("europe_north_b", "us_east_a"),
    matrix_halves=m.get(n.rtt_pair_key("europe_north_a", "europe_north_b")),
    lookup_sibling=_lookup_rtt_ms(m, "europe_north_a", "DK_East"),
    lookup_own=_lookup_rtt_ms(m, "europe_north_a", "Belgium"),
    lookup_remote=_lookup_rtt_ms(m, "europe_north_a", "BR_South"),
)))
"""


def probe(topology_path):
    env = dict(os.environ, SIM_TOPOLOGY_FILE=str(topology_path),
               PYTHONPATH=f"{ROOT}:{ROOT / 'proto'}")
    p = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def with_topology(edit):
    t = json.load(open(SAME16))
    edit(t)
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(t, f)
    f.close()
    return f.name


def test_split_owns_and_prices_as_parent():
    rc, out, err = probe(SAME16)
    assert rc == 0, err
    r = json.loads(out)
    assert r["group"] == {"Netherlands": "europe_north_a", "DK_East": "europe_north_b",
                          "BR_North": "south_america_b", "Chile": "south_america_a"}, r["group"]
    assert r["owns"] == {"europe_north_a": ["Netherlands", "Belgium"],
                         "europe_north_b": ["DK_East", "DK_West"]}, r["owns"]
    assert r["rtt_halves"] == 2.0, r["rtt_halves"]           # europe_north's own round trip
    assert r["rtt_sa_halves"] == 10.0, r["rtt_sa_halves"]     # south_america's own round trip
    assert r["rtt_cross"] == 85.0, r["rtt_cross"]             # europe_north <-> us_east
    assert r["matrix_halves"] == 2.0, r["matrix_halves"]
    assert r["lookup_sibling"] == 2.0, r["lookup_sibling"]    # the other half is a 2 ms hop
    assert r["lookup_own"] == 0.0, r["lookup_own"]
    assert r["lookup_remote"] == 200.0, r["lookup_remote"]    # europe_north <-> south_america


def test_every_dc_owned_once():
    t = json.load(open(SAME16))
    owned = [dc for dcs in t["topology"].values() for dc in dcs]
    assert sorted(owned) == sorted(json.load(open(BASE))["all_dcs"]), owned
    assert len(owned) == len(set(owned)) == 16


def test_stray_dc_refused():
    path = with_topology(lambda t: t["topology"]["europe_north_a"].append("Chile"))
    rc, _, err = probe(path)
    assert rc != 0 and "not in 'europe_north'" in err, err


def test_parent_cannot_also_run():
    path = with_topology(lambda t: t["topology"].update(europe_north=["Netherlands"]))
    rc, _, err = probe(path)
    assert rc != 0 and "cannot also run" in err, err


def test_no_split_changes_nothing():
    code = ("import json, common.network as n; print(json.dumps([n.GROUP_PARENT, "
            "n.REGION_TO_GROUP['DK_East'], n.get_rtt_ms('europe_north', 'us_east')]))")
    env = dict(os.environ, SIM_TOPOLOGY_FILE=str(BASE), PYTHONPATH=f"{ROOT}:{ROOT / 'proto'}")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == [{}, "europe_north", 85.0], p.stdout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

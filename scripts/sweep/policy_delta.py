#!/usr/bin/env python3
"""Policies on the two axes that decide them: carbon saved and SLO met.

    PYTHONPATH=.:proto TTFT_BUDGET=3000 TPOT_BUDGET=200 \\
        python3 scripts/sweep/policy_delta.py facility_ilp_carbon_sla \\
            ilp_carbon_sla iterative_ilp_carbon_sla <root>...

Any number of policies, then the roots (an argument that is an existing
directory starts the roots). Per root and rate: each policy's carbon saving
against round_robin in the same cell and its SLO, then the deltas against the
FIRST policy listed. A policy only wins if it takes the first axis without
giving back the second, so both are needed to read a cell.

`WHY=1` adds the work/idle carbon split, which says HOW the gap arose: `work`
means A placed the same requests on cleaner sites, `idle` means A left fewer --
or cleaner -- sites powered and empty.

TPOT_BUDGET must match the run (200 for Mixtral, 100 for Llama/Qwen) or the
SLO columns are scored against the wrong budget.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "proto"))

# By path, not as a package: scripts/ has no __init__.py and adding one to
# import a sibling would make every other script in it importable by accident.
import importlib.util as _il
_spec = _il.spec_from_file_location(
    "carbon_slo_tables",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "carbon_slo_tables.py"))
_cst = _il.module_from_spec(_spec)
_spec.loader.exec_module(_cst)
read_energy, read_slo = _cst.read_energy, _cst.read_slo

def main(argv):
    pols = []
    for i, arg in enumerate(argv):
        if Path(arg).is_dir():
            pols, roots = argv[:i], argv[i:]
            break
    else:
        pols, roots = [], []
    if len(pols) < 2 or not roots:
        print(__doc__)
        return 1
    ttft = float(os.environ.get("TTFT_BUDGET", "3000"))
    tpot = float(os.environ.get("TPOT_BUDGET", "100"))
    why = os.environ.get("WHY", "") not in ("", "0", "false")
    short = [p.replace("_carbon_sla", "").replace("carbon_aware_a", "ca")[:9]
             for p in pols]
    hdr = f"{'rate':>6} " + "".join(f"{n+' save':>11}{'SLO':>7}" for n in short)
    hdr += "".join(f"{'d save':>9}{'d SLO':>8}" for n in short[1:])
    if why:
        hdr += "".join(f"{'d work g':>10}{'d idle g':>10}" for n in short[1:])
    for root in roots:
        print(f"######## {root}   " + "  ".join(f"{n}={p}" for n, p in zip(short, pols)))
        print(hdr)
        rates = sorted({int(p.name[5:]) for p in Path(root, pols[0]).glob("rate_*")
                        if p.name[5:].isdigit()})
        for rate in rates:
            cells = []
            for pol in list(pols) + ["round_robin"]:
                d = Path(root, pol, f"rate_{rate}")
                en, res = d / "energy.json", d / "results.csv"
                if not (en.exists() and res.exists()):
                    cells = []
                    break
                cells.append((read_energy(en), read_slo(res, ttft, tpot)))
            if len(cells) != len(pols) + 1 or any(s is None for _, s in cells):
                continue
            rr = cells[-1][0]
            sav = lambda c: (100.0 * (1 - c["total"] / rr["total"])
                             if rr["total"] else float("nan"))
            line = f"{rate:>6} " + "".join(
                f"{sav(c):>10.1f}%{s['ok']:>6.1f}%" for c, s in cells[:-1])
            (c0, s0) = cells[0]
            for c, s in cells[1:-1]:
                line += f"{sav(c)-sav(c0):>+9.1f}{s['ok']-s0['ok']:>+8.1f}"
            if why:
                for c, _ in cells[1:-1]:
                    line += (f"{c['work']-c0['work']:>+10.1f}"
                             f"{c['idle']-c0['idle']:>+10.1f}")
            print(line)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

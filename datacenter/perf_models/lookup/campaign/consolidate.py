"""Consolidate per-group CSVs into per-BS CSVs.

After a drain sweep finishes, each group has its own directory with
per_step.csv and per_request.csv. This helper walks those directories
and writes two summary files per BS value:

    per_request_bs{BS}.csv   all requests across all groups of this BS
    per_step_bs{BS}.csv      all steps across all groups of this BS

Both files get an added `group_idx` column so you can trace any row
back to its source group.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


GROUP_RE = re.compile(r"group_bs(\d+)_(\d+)")


def consolidate(out_root: Path) -> None:
    """Walk all group_bs*_* subdirs under out_root and write per-BS CSVs."""
    out_root = Path(out_root)
    if not out_root.exists():
        return

    # Collect all group dirs indexed by (bs, group_idx)
    groups_by_bs: Dict[int, List[tuple]] = defaultdict(list)
    for entry in sorted(out_root.iterdir()):
        if not entry.is_dir():
            continue
        m = GROUP_RE.match(entry.name)
        if not m:
            continue
        bs = int(m.group(1))
        gidx = int(m.group(2))
        groups_by_bs[bs].append((gidx, entry))

    for bs, groups in groups_by_bs.items():
        groups.sort()

        # ── per_request consolidation ──
        pr_out = out_root / f"per_request_bs{bs}.csv"
        pr_rows: List[dict] = []
        pr_header: List[str] = []
        for gidx, gdir in groups:
            pr_file = gdir / "per_request.csv"
            if not pr_file.exists():
                continue
            with open(pr_file, newline="") as f:
                reader = csv.DictReader(f)
                if not pr_header:
                    pr_header = ["bs", "group_idx"] + (reader.fieldnames or [])
                for row in reader:
                    out_row = {"bs": bs, "group_idx": gidx}
                    out_row.update(row)
                    pr_rows.append(out_row)
        if pr_rows:
            with open(pr_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pr_header)
                w.writeheader()
                w.writerows(pr_rows)

        # ── per_step consolidation ──
        ps_out = out_root / f"per_step_bs{bs}.csv"
        ps_rows: List[dict] = []
        ps_header: List[str] = []
        for gidx, gdir in groups:
            ps_file = gdir / "per_step.csv"
            if not ps_file.exists():
                continue
            with open(ps_file, newline="") as f:
                reader = csv.DictReader(f)
                if not ps_header:
                    ps_header = ["bs", "group_idx"] + (reader.fieldnames or [])
                for row in reader:
                    out_row = {"bs": bs, "group_idx": gidx}
                    out_row.update(row)
                    ps_rows.append(out_row)
        if ps_rows:
            with open(ps_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ps_header)
                w.writeheader()
                w.writerows(ps_rows)

        print(f"[consolidate] BS={bs}: {len(pr_rows)} requests, "
              f"{len(ps_rows)} steps → per_request_bs{bs}.csv, "
              f"per_step_bs{bs}.csv")

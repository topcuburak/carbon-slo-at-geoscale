"""Load-dependent RTT: the one place a datacentre's occupancy becomes latency.

The model turns how busy a group of sites is into how slow the paths to and
from it are, so that load shows up in TTFT rather than only in queueing. It is
a modelling choice and not a measurement: a network quantity derived from a
compute one, on the reasoning that a congested region's paths are congested
too.

Two things worth knowing before reading a number out of it.

The eight same-group pairs are computed and never used. A scheduler placing
inside its own group short-circuits to 0 before it reads the matrix
(`_lookup_rtt_ms`), so only the 28 cross-group pairs reach a decision, and any
figure quoted about this model should be over those.

And the normalisation decides whether the model responds to load at all. See
`CongestionModel`.
"""

import logging
from typing import Dict

import inference_pb2

from common.network import (
    REGION_TO_GROUP,
    get_all_base_rtts,
    get_all_group_names,
)

log = logging.getLogger("provisioner.congestion")


class CongestionModel:
    """Scales the base RTT matrix by how loaded each scheduler group is.

    `norm` is the interesting parameter.

      "cap"   heat is measured against a fixed per-engine batch, so an idle
              fleet inflates nothing and a saturated one reaches the beta
              ceiling. Load moves the matrix.
      "peak"  heat is measured against the busiest group, which makes that
              group 1.0 by construction. Measured over the 28 cross-group
              pairs at beta 0.5, all of 1, 8, 64, 128 and 256 requests per
              datacentre give the SAME +50% and the same +140 ms worst case:
              four orders of magnitude of load, one matrix. What it measures
              is the spread between groups, not the level. This is the
              behaviour every result before 2026-09-02 was produced under.

    Neither changes WHERE requests go under uniform load -- measured, both
    reorder destinations for 0 of 8 source groups -- because scaling every path
    equally preserves order. They differ in whether the number responds to load.
    """

    def __init__(self, beta: float = 0.5, norm: str = "cap",
                 heat_cap: float = 32.0) -> None:
        if norm not in ("cap", "peak"):
            raise ValueError(f"norm must be 'cap' or 'peak', got {norm!r}")
        self.beta = float(beta)
        self.norm = norm
        # The per-engine batch that counts as fully loaded. Should track the
        # engines' max_num_seqs -- the datacentres do not report it, so it is
        # set here and the sweep must keep the two in step.
        self.heat_cap = float(heat_cap) if heat_cap > 0 else 32.0

    def group_heat(self, dc_view: Dict[str, inference_pb2.DcStats]) -> Dict[str, float]:
        """Mean per-engine batch over each group's datacentres.

        A plain mean, so it is insensitive to group size: one overloaded site
        moves 7-site `middle_east` far more than 26-site `asia`. And a group
        with no reachable sites reads 0.0 -- "empty" where it means "unknown".
        Both are recorded in SIMULATOR_TODO item 16 and neither is measured.
        """
        groups = get_all_group_names()
        loads: Dict[str, list] = {g: [] for g in groups}
        for dc_id, stats in dc_view.items():
            g = REGION_TO_GROUP.get(dc_id)
            if g is not None:
                loads[g].append(stats.active_requests / max(stats.num_gpus, 1))
        return {g: (sum(v) / len(v) if v else 0.0) for g, v in loads.items()}

    def normalise(self, raw_heat: Dict[str, float]) -> Dict[str, float]:
        groups = get_all_group_names()
        if self.norm == "peak":
            mx = max(raw_heat.values()) if raw_heat else 0.0
            return {g: (raw_heat[g] / mx if mx > 0 else 0.0) for g in groups}
        return {g: min(1.0, raw_heat[g] / self.heat_cap) for g in groups}

    def rtt_matrix(self, dc_view: Dict[str, inference_pb2.DcStats]) -> Dict[str, float]:
        """dynamic(A,B) = base(A,B) * (1 + beta * (norm[A] + norm[B]) / 2)."""
        norm_heat = self.normalise(self.group_heat(dc_view))
        out: Dict[str, float] = {}
        for key, base in get_all_base_rtts().items():
            g1, g2 = key.split("|")
            path_heat = (norm_heat[g1] + norm_heat[g2]) / 2.0
            out[key] = base * (1.0 + self.beta * path_heat)
        return out

"""Power usage effectiveness, per datacentre.

PUE is the ratio of what a facility draws to what its IT equipment draws:
cooling, power conversion and everything else on the site, divided into the
compute. It multiplies every joule on the way to carbon, so it is not a
detail -- a site assumed at 1.1 that is really 1.5 under-reports its emissions
by 36%, and because that error is a per-site multiplier it does NOT cancel when
two policies are compared on sites with different values.

The problem formulation indexes it by site, PUE_i, and facilities genuinely
differ: hyperscale operators report annual figures near 1.1, while colocation
and older enterprise rooms are commonly quoted at 1.4-1.8. The code used to
carry a single global constant, so the index in the formulation described
something the simulator did not do.

Values live in `common/pue_by_dc.json`, keyed by the datacentre names in
`common/network.py:SCHEDULER_GROUPS`. Anything absent from the file falls back
to `DEFAULT_PUE`.

**The shipped file sets every datacentre to the default.** That is deliberate,
not an oversight: real per-site PUE has to come from a source that can be
cited, and inventing plausible-looking per-zone numbers would put fabricated
data underneath a carbon result. The mechanism is here so that measured values
can be dropped in and so a sensitivity study can vary them; until they are, the
runs are explicitly uniform and should be described that way.

    from common.pue import get_pue
    pue = get_pue("Ireland")          # -> DEFAULT_PUE unless the file says otherwise
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_PUE", "get_pue", "load_pue_table", "pue_table",
           "PUE_TABLE_PATH"]

# The uniform value every published run so far used. Keep it as the fallback so
# a datacentre missing from the table behaves exactly as it did before.
DEFAULT_PUE: float = float(os.environ.get("DEFAULT_PUE", "1.1"))

PUE_TABLE_PATH: str = os.environ.get(
    "PUE_TABLE",
    str(Path(__file__).resolve().parent / "pue_by_dc.json"),
)

_TABLE: Dict[str, float] | None = None


def load_pue_table(path: str | None = None, force: bool = False) -> Dict[str, float]:
    """Read the per-datacentre table, once, and cache it.

    A missing or unreadable file is not an error -- it means every site takes
    the default, which is the behaviour this replaced. A malformed one is
    logged rather than raised, because a carbon run should not die on a
    configuration file it can fall back from.
    """
    global _TABLE
    if _TABLE is not None and not force and path is None:
        return _TABLE
    p = path or PUE_TABLE_PATH
    table: Dict[str, float] = {}
    try:
        with open(p) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k.startswith("_"):          # _comment, _default, _source
                continue
            table[k] = float(v)
    except FileNotFoundError:
        log.debug("no PUE table at %s; every site takes %.3f", p, DEFAULT_PUE)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        log.warning("PUE table at %s is unreadable (%s); every site takes "
                    "%.3f", p, e, DEFAULT_PUE)
    if path is None:
        _TABLE = table
    return table


def pue_table() -> Dict[str, float]:
    """The loaded table. Empty means every site takes the default."""
    return dict(load_pue_table())


def get_pue(dc_id: str) -> float:
    """This datacentre's PUE, or the default if the table does not name it."""
    return load_pue_table().get(dc_id, DEFAULT_PUE)

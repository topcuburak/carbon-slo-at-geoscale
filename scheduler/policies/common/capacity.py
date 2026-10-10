"""How much a datacentre may hold, and how much room is left in it.

`bs_cap` is a herd guard, not a capacity model. It exists so that a stale
snapshot cannot let several schedulers pile onto one site before any of them
sees the others' placements: it caps what a site is allowed to be carrying at
the moment a decision is taken.

**It is PER ENGINE, and the comparison scales it by the site's engine count.**

That is not how it started. Every reader of `bs_cap` used to compare it against
`DcStats.active_requests + pending_requests`, which are sums over the whole
site, so the number's unit depended on the topology and the name said nothing
about it. Setting it to `max_num_seqs` -- the obvious thing to do, given the
name -- throttled a site to 1/engines of its real capacity, silently:

    8 sites, 8 engines each, max_num_seqs=128 -> real capacity 1024 per site
    bs_cap=128 compared site-wide      -> 128, an eighth of that
    every site sat at a median 217-327 in flight, so every site read as full
    the LP could seat nothing and every request fell to the round-robin rung
    ilp_carbon_sla then placed 52% of requests on the clean sites, against
    round_robin's 50%: the carbon objective never ran, and nothing said so

Per engine, `bs_cap=128` on that fleet means 1024 site-wide, which is exactly
what the engines will admit -- the guard tracks the hardware instead of
fighting it. A tighter guard is expressed as a smaller per-engine number, and
the same config now means the same thing on a one-node site and a four-node
one.

Note what this does NOT change: the guard is still cruder than the SLA filter,
which is the mechanism actually meant to keep placements inside their latency
budget. `bs_cap=0` turns the guard off and leaves the SLA filter alone, and
that is a reasonable way to run.
"""

from __future__ import annotations

from scheduler.policies.common.carbon_signal import effective_carbon_intensity

__all__ = ["dc_engines", "dc_load", "dc_capacity", "at_cap", "dc_headroom",
           "greenest_under_cap"]


def dc_engines(stats) -> int:
    """Engines at this site. `num_gpus` counts engines, not physical cards."""
    return max(int(getattr(stats, "num_gpus", 1) or 1), 1)


def dc_load(stats) -> int:
    """What the site is carrying: active plus pending, over all its engines.

    The two are disjoint in gpu_sim -- `_pending_count` is decremented before
    `_active_count` is incremented -- so the sum is the whole in-flight
    population and counts nothing twice.
    """
    return (int(getattr(stats, "active_requests", 0))
            + int(getattr(stats, "pending_requests", 0)))


def dc_capacity(stats, bs_cap_per_engine: int) -> int:
    """The site-wide seat count `bs_cap_per_engine` describes. 0 means no cap."""
    if not bs_cap_per_engine or bs_cap_per_engine <= 0:
        return 0
    return int(bs_cap_per_engine) * dc_engines(stats)


def at_cap(stats, bs_cap_per_engine: int) -> bool:
    """True when the site is at or over the guard and must not take more."""
    cap = dc_capacity(stats, bs_cap_per_engine)
    return cap > 0 and dc_load(stats) >= cap


def dc_headroom(stats, bs_cap_per_engine: int):
    """Seats the guard still allows, or None when there is no guard.

    None and 0 are different answers and callers must keep them apart: None is
    "unconstrained", 0 is "constrained and full". Collapsing them is how a
    disabled cap turns into a cap of zero.
    """
    cap = dc_capacity(stats, bs_cap_per_engine)
    if cap <= 0:
        return None
    return max(0, cap - dc_load(stats))


def greenest_under_cap(dc_view, bs_cap_per_engine: int):
    """Lowest carbon intensity among sites still under the guard, or None.

    The rung between "nothing met the SLA" and a carbon-blind fallback. The SLA
    filter is an ESTIMATE, and it is the thing that just rejected everything;
    `bs_cap` is a counted fact about load. Dropping the estimate while keeping
    the count preserves the policy's whole purpose without piling onto a site
    that is already full.

    Round-robin was chosen as the fallback originally, over least_loaded, on
    the grounds that a carbon policy's loaded sites ARE its green ones, so
    least_loaded reliably picks the carbon-expensive one and turns an SLA
    failure into a carbon failure. That holds against least_loaded and not
    against this. Measured at 3200 req/s with BS_CAP=20, about 8% of requests
    reached the round-robin rung, and that 8% alone accounts for the whole of
    the carbon lost down the cap axis: 46.6% saving against 42.5%, where
    0.92 x 46.6 = 42.9%.

    With bs_cap disabled (0) `at_cap` is always False, so this is simply the
    greenest site -- which is the right answer when there is no load guard to
    respect.
    """
    best, best_ci = None, None
    for dc_id, stats in dc_view.items():
        if at_cap(stats, bs_cap_per_engine):
            continue
        ci = effective_carbon_intensity(stats)
        if best_ci is None or ci < best_ci:
            best, best_ci = dc_id, ci
    return best

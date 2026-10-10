"""Fold placements a scheduler has made into a datacentre's counters.

The provisioner's snapshot describes a datacentre as it was when it was last
polled. Anything a scheduler has placed since then is invisible to it, so a
decision taken later in the same period prices a datacentre that is emptier
than the real one. This is the correction, and it is deliberately ONE function:
the rule it encodes is subtle enough that a second copy would drift.

Two consumers, the same arithmetic:

  iterative_ilp_carbon_sla   between its own rounds -- round n+1 prices the
                             view round n's assignment produced.
  scheduler/server.py        for requests forwarded IN from another scheduler,
                             which land without this scheduler deciding
                             anything and would otherwise not appear until the
                             next poll.

The invariants worth knowing before changing anything here:

  admitted vs queued   A placement at a datacentre with no free slot does not
                       start running. The engine holds it on its semaphore,
                       where it is pending, not active. Projecting every
                       placement as running understates the queue whoever reads
                       this next prices their TTFT against.

  active/pending are disjoint   the engine decrements _pending_count before
                       it increments _active_count, so active + pending is the
                       whole in-flight population. A placement is added to
                       exactly ONE of them; adding it to both showed 2n of load
                       for n placements.

  THREE stages, not two   this is the invariant the first version of this
                       function got wrong, and it cost the TTFT estimate most
                       of its accuracy. A request at an engine is queued (on
                       the semaphore), then PREFILLING (admitted, holding a
                       slot, prompt not yet through a forward pass), then
                       decoding. inference_engine.py admits at ~1163 with
                       `_prefill_count += 1; _total_prefill_isl += in_tok` and
                       touches NEITHER `_decoding_count` NOR
                       `_total_active_isl`; those move only when the prefill
                       step completes, at ~1060-1083. So a placement that is
                       admitted is PREFILLING. Projecting it as decoding was
                       wrong twice over:

                         - `estimate_ttft_ms` prices a slot wait as
                           (R/d) * step / d, quadratic in d, so every phantom
                           decoder collapsed the admission term. Measured over
                           probe_rate2, the estimate charged 0-17% of the seat
                           wait the run actually billed.
                         - the fused-prefill term reads total_prefill_isl, so
                           a placement left out of it made every arrival
                           believe it prefilled ALONE. Measured, the assumed
                           fused batch was 1 against a real 11.

  the three ISL sums are disjoint   a prompt is in exactly one of
                       total_pending_isl (queued), total_prefill_isl
                       (admitted, unprefilled) or total_active_isl (decoding,
                       and thus KV the batch holds). Counted in two, a
                       placement inflates TPOT and TTFT at the same time.

  remaining_osl follows the admitted set   a prefilling request still owes its
                       whole output, so it is charged here even though it is
                       not yet decoding. Left un-projected, the batch grows
                       while its remaining work stands still, which reads as a
                       batch about to drain.
"""

from __future__ import annotations

from typing import Optional, Sequence

import inference_pb2

__all__ = ["project_placements"]


def project_placements(
    stats: "inference_pb2.DcStats",
    isls: Sequence[int],
    osls: Sequence[int],
    max_num_seqs: int,
    stages: Optional[Sequence[str]] = None,
) -> "inference_pb2.DcStats":
    """`stats` with `len(isls)` placements folded in. Never mutates `stats`.

    isls and osls are per placement and positionally paired; osl is whatever
    OSL_SOURCE resolved upstream (oracle, predicted or ref), not the ground
    truth, unless the run is configured for oracle.

    max_num_seqs is the engines' per-engine limit; the datacentre's capacity is
    that times its engine count. 0 means unknown, and then every placement is
    projected as admitted -- the pre-cap behaviour.

    stages says which lifecycle stage each placement is ALREADY in: "f" for
    one that has not finished its prompt, "d" for one that is decoding. None
    means every placement is brand new, which is the right default for a
    caller projecting assignments it has not dispatched yet.

    WHY THE STAGE MATTERS, measured. Projecting a whole in-flight population
    into ONE stage is worse than not projecting at all. Scored offline over
    probe_rate2's decision logs, est/measured TTFT at p50:

        arm                              rr/400   ll/800   ca/1200
        no projection                     1.66     0.32      0.62
        all as decoding (the old bug)     1.67     0.33      0.82
        all as prefilling                 4.80     1.18      3.52
        by actual stage                   2.45     1.02      1.42

    A request dispatched two seconds ago is not prefilling any more, and
    charging its prompt to an arrival's fused pass invents work that has
    already happened.
    """
    n_assigned = len(isls)
    if stages is None:
        stages = ("f",) * n_assigned

    # Already decoding: past admission, so they do not compete for a slot.
    dec_isl = sum(i for i, st in zip(isls, stages) if st == "d")
    dec_osl = sum(o for o, st in zip(osls, stages) if st == "d")
    n_dec = sum(1 for st in stages if st == "d")

    # Not yet through their prompt: these are what a new arrival queues behind.
    new_isls = [i for i, st in zip(isls, stages) if st != "d"]
    new_osls = [o for o, st in zip(osls, stages) if st != "d"]
    n_new = len(new_isls)
    assigned_isl_sum = sum(new_isls)

    return project_totals(stats, n_new, assigned_isl_sum, sum(new_osls),
                          n_dec, dec_isl, dec_osl, max_num_seqs)


def project_totals(
    stats: "inference_pb2.DcStats",
    n_new: int, isl_new: float, osl_new: float,
    n_dec: int, dec_isl: float, dec_osl: float,
    max_num_seqs: int,
) -> "inference_pb2.DcStats":
    """The projection above, from TOTALS instead of per-placement lists.

    Nothing here ever needed the individual placements -- six sums are the
    whole input -- and building the lists to produce them is what made the
    scheduler's per-request view rebuild cost O(in-flight). A caller that can
    maintain the sums incrementally calls this directly; project_placements
    keeps the list interface for callers that cannot, such as the ILP folding
    in one round's assignments before the next.
    """
    assigned_isl_sum = isl_new
    engines = max(int(getattr(stats, 'num_gpus', 1) or 1), 1)
    in_flight = (int(stats.active_requests)
                 + int(stats.pending_requests))
    cap = int(max_num_seqs or 0) * engines
    free = max(0, cap - in_flight - n_dec) if cap > 0 else n_new
    admitted = min(n_new, free)
    queued = n_new - admitted

    # Split their token sums the same way, pro rata: this has counts, not which
    # individual request landed where.
    share = (admitted / n_new) if n_new else 0.0
    isl_admitted = assigned_isl_sum * share
    isl_queued = assigned_isl_sum - isl_admitted

    proj = inference_pb2.DcStats()
    proj.CopyFrom(stats)
    # active covers both stages, so it takes the admitted share whichever one
    # they are in.
    proj.active_requests = stats.active_requests + admitted + n_dec
    # Admitted means PREFILLING, not decoding -- see the three-stage invariant
    # above. decoding_requests and total_active_isl are deliberately left
    # alone: nothing decodes until a prefill pass has run.
    proj.prefilling_requests = (
        int(getattr(stats, 'prefilling_requests', 0)) + admitted)
    proj.total_prefill_isl = int(
        int(getattr(stats, 'total_prefill_isl', 0)) + isl_admitted)
    # Owed output, charged on admission: the request will decode all of it.
    proj.total_remaining_osl = int(
        int(getattr(stats, 'total_remaining_osl', 0))
        + osl_new * share + dec_osl)
    # Past their prompt already: decoding, with their prompts in the KV the
    # batch holds rather than in the prefill still owed.
    proj.decoding_requests = (
        int(getattr(stats, 'decoding_requests', 0)) + n_dec)
    proj.total_active_isl = int(
        int(getattr(stats, 'total_active_isl', 0)) + dec_isl)
    # Still on the semaphore.
    proj.pending_requests = stats.pending_requests + queued
    proj.total_pending_isl = int(
        int(getattr(stats, 'total_pending_isl', 0)) + isl_queued)
    # Every prompt sent since the snapshot takes cache room ahead of the next
    # one, admitted or not: its share of one engine's; see DcStats.kv_queue_ms.
    proj.kv_room_tokens = (
        float(getattr(stats, 'kv_room_tokens', 0.0) or 0.0)
        - assigned_isl_sum / engines)
    return proj

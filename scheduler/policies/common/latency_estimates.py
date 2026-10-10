"""Per-DC TTFT and TPOT estimates from a provisioner snapshot.

One copy. These lived inline in three policies -- ilp_carbon,
greedy.carbon_aware and greedy.carbon_aware_local -- as three identical
bodies, so a correction to one silently left the other two on the old model
and the difference then read as "greedy is worse than ILP" for a reason that
was not real.

WHAT THESE ARE FOR. They price a placement decision, not a measurement. The
simulator's own timings are the ground truth reported in results.csv; these
only decide where a request goes and whether a site passes its SLA check.
An error here costs policy quality, not reporting validity.

HOW GOOD THEY CAN BE. Not very, per request, and that is structural. The
scheduler sees a snapshot at most one poll period old. Scored on 2676
decisions from an overloaded 32-DC run, requests that arrived under the SAME
snapshot at the SAME datacentre -- placements the policy cannot tell apart --
differed in outcome by 150% of their median at p50 and 1079% at p90; one
snapshot served 111 requests whose waits ran from 9.6 s to 22.3 s. No
estimator reading that snapshot can score better than that spread. So the
target is the CENTRE of the distribution, not its width: median and mean
ratio against measured, with MAPE reported only as context.

MEASURED, ratio of predicted to measured (queue_ms + ttft_ms), on two arms.
Read them together: the estimate is centred at moderate load and still under
at overload, where the backlog outgrows any snapshot the scheduler was given.

                                      overloaded          moderate
                                    (rate 120, 2676)   (rate 25, 479)
    form                             p50   mean         p50   mean
    prefill-only drain (was)        0.29   0.47        0.81   0.73
    + admission + step boundary     0.72   0.89        1.01   0.97

Reproduce with scheduler/score_ttft_estimator.py against a DECISION_LOG; its
--spread flag prints the floor described above for whatever arm you scored.

Two terms were missing, both real:

1. THE ADMISSION WAIT. An engine caps queue plus running at max_num_seqs. At
   the cap it admits nothing until a decoding request emits its last token, so
   a new arrival waits for sequence slots to free, not for the prefill ahead
   of it. That wait is set by how much decode work the batch still owes --
   DcStats.total_remaining_osl -- and the old model priced it at zero. It is
   the dominant term under load: measured queue_ms alone was 4449 ms at p50 in
   that run against a total of 4858 ms.

   Gated on the engine's max_num_seqs, and the gate is NOT what the scores
   argued for. On real snapshots the two forms were indistinguishable -- the
   term is proportional to queue depth, so it already vanishes where a gate
   would switch it off (below the cap, 0.50 gated against 0.57 ungated at
   p50). The gate is there for a case the scored data does not contain: the
   ILP projects its own in-flight assignments into pending_requests, and those
   requests are not queued behind a full engine, they are in flight. Ungated,
   that projection charged a slot wait to a site with free slots and pushed it
   out of the feasible set -- an eight-engine site with 64 decoders, no real
   queue and a 3 s budget scored 4.6 s. With max_num_seqs unknown the gate is
   off, which is the behaviour that was scored.

2. THE STEP BOUNDARY. The engine runs one step at a time and decides what to
   run next only at a step boundary, so an arrival waits out the decode step
   in progress before its prompt can be prefilled. Measured directly: on
   snapshots with an empty queue the residual against the old model was 17-21
   ms, flat in ISL, against a decode step of 19.4 ms at those same snapshots.

What remains is a 28% median under-estimate, and the snapshot spread above is
the reason to stop rather than add terms: the next candidate is compensating
for staleness with the requests this scheduler has itself dispatched since the
poll, which is a different fix with different bookkeeping.
"""


#: Where a placement decision gets the request's output length. It enters the
#: carbon score linearly (energy is per decode step), the keep-alive term
#: non-linearly, and the latency score as the multiplier on TPOT.
#:
#:   "oracle"     the trace's out_tok -- the number the datacentre will actually
#:                generate. This is what shipped, and it makes the scheduler
#:                clairvoyant. Fine as a stated upper bound, not as the only
#:                configuration a paper reports.
#:   "predicted"  what a length predictor said, carried on the request. On the
#:                trace that has both, the predictor's mean is 913 tokens
#:                against a true 542 -- +69% bias, 38.5% median absolute error.
#:   "ref"        a flat constant for every request, which is what the greedy
#:                latency score used regardless of the other two.
#:
#: A uniform scaling of OSL largely cancels in a per-request argmin over
#: datacentres -- the same factor multiplies every candidate -- so the oracle
#: buys the greedy policies little. It does NOT cancel in the ILP, which
#: assigns a batch of requests across sites and therefore depends on their
#: RELATIVE lengths, nor in the keep-alive term, which compares this request's
#: length against the batch's remaining work.
import os

OSL_SOURCES = ("oracle", "predicted", "ref")

#: Ceiling on the KV gate's wait in a TTFT estimate, ms. Past any budget by an
#: order of magnitude; it only keeps a site whose cache frees nothing finite, so
#: score normalisation over the candidates still has a maximum.
KV_WAIT_CAP_MS = 60_000.0

#: Whose TPOT the feasibility test is about.
#:
#: "arrival" (default, and what every result so far was measured under) asks
#: what the request being placed would experience there.
#:
#: "batch" asks what the batch ALREADY RUNNING at that datacentre will
#: experience once this request's prompt lands on it. Two fields change:
#:
#:   batch    active_requests + pending_requests, not decoding_requests. The
#:            view projection never advances decoding_requests -- nothing
#:            decodes before its prefill runs -- so under "arrival" a
#:            scheduler's own dispatches cannot move the batch term and it
#:            cannot see itself filling the site.
#:   backlog  total_prefill_isl + total_pending_isl, not the admitted set
#:            alone. Both get prefilled while the running batch decodes, and
#:            both block it while they do.
#:
#: Nothing else moves: the horizon and the duty divisor are the shipped ones.
TPOT_FILTER = os.environ.get("TPOT_FILTER", "arrival").strip().lower()


def resolve_osl(true_osl: int, predicted_osl: int, source: str,
                ref_osl: int = 500) -> int:
    """The output length a placement decision should be priced on.

    Returns (value, fell_back) so a caller can report how often the requested
    source was unavailable rather than let a mixed population pass silently.
    """
    if source == "predicted":
        return (int(predicted_osl), False) if predicted_osl > 0 else (int(ref_osl), True)
    if source == "ref":
        return int(ref_osl), False
    return (int(true_osl), False) if true_osl > 0 else (int(ref_osl), True)


def _per_engine(stats):
    """Snapshot quantities divided down to one engine.

    Everything a datacentre reports is a fleet-wide sum, and its arrivals are
    round-robined across engine slots that drain in parallel, so what a new
    request inherits is one engine's share. Pricing the fleet-wide backlog as
    a single fused pass overstated the drain by the engine count -- 7.9x on an
    eight-engine site -- which pushed loaded sites out of the feasible set.
    """
    e = max(int(getattr(stats, 'num_gpus', 1)), 1)
    # Decoding requests, not active ones: the two differ by whatever is still
    # waiting on its prompt, and those emit no token in the step being priced.
    # It also has to match the sums below, which are over the decoding set
    # only. Falls back to active_requests for a datacentre predating the field.
    decoding = (int(getattr(stats, 'decoding_requests', 0))
                or int(getattr(stats, 'active_requests', 0)))
    return {
        "engines": e,
        "d": decoding / e,
        # Everything admitted, whatever stage: decoding plus prefilling, and
        # more besides -- a request past its prefill but not yet in the decode
        # loop is in neither of those two counts, so this is not d + f.
        "a": int(getattr(stats, 'active_requests', 0)) / e,
        # Admitted but not yet prefilled: holds a slot, contributes no token
        # yet, and is what a new arrival's own prefill fuses with.
        "f": int(getattr(stats, 'prefilling_requests', 0)) / e,
        "prefill_isl": int(getattr(stats, 'total_prefill_isl', 0)) / e,
        "p": int(getattr(stats, 'pending_requests', 0)) / e,
        "remaining_osl": int(getattr(stats, 'total_remaining_osl', 0)) / e,
        "pending_isl": int(getattr(stats, 'total_pending_isl', 0)) / e,
        "kv": (int(getattr(stats, 'total_active_isl', 0))
               + int(getattr(stats, 'total_generated_tokens', 0))) / e,
        # Share of engine wall time with decode BLOCKED by a prefill pass.
        # Already a per-engine fraction, so it is NOT divided by e.
        "duty": float(getattr(stats, 'prefill_duty', 0.0) or 0.0),
    }


def estimate_tpot_ms(stats, analytical_model, bs_offset: int = 1,
                     req_osl: int = 0, req_isl: int = 0) -> float:
    """Per-token decode time this request would see.

    bs_offset is what the caller adds to the current batch: 1 for a greedy
    policy placing one request, 0 for the ILP, which projects the batch itself.

    req_osl is what the placement is priced on (see resolve_osl). It sets the
    HORIZON, and the horizon is the whole second half of this estimate -- see
    the queued-prefill term below. 0 leaves that term out, which is the
    behaviour before it existed.
    """
    q = _per_engine(stats)
    seated = q["a"] + q["p"] if TPOT_FILTER == "batch" else q["d"]
    projected_bs = max(1, (seated * q["engines"] + bs_offset) / q["engines"])
    step_ms = analytical_model.tpot_ms(int(projected_bs), total_kv=int(q["kv"]))

    # tpot_ms prices the DECODE STEP. What the request actually experiences is
    # the step plus every stall in between: the engine blocks all decode while
    # a prefill forward pass runs (see datacenter/inference_engine.py), so a
    # site spending a fraction `duty` of its time prefilling stretches every
    # token by 1/(1 - duty).
    #
    # Measured 2026-09-04 over probe_rate2, round_robin at 8 sites x 16
    # engines, snapshots ~1 s old so staleness is not in play:
    #
    #     rate   duty   m_tpot/est_tpot   after 1/(1-duty)
    #      400  0.139         1.25x             1.07x
    #      800  0.277         1.70x             1.23x
    #     1200  0.416         2.65x             1.55x
    #
    # Without this the SLA filter is fed a number that is optimistic by more
    # the busier the site is, so it never rejects a saturated one -- which is
    # how carbon_aware came to pack two sites to the batch cap and violate the
    # TPOT budget on 74% of its requests while its estimate said 15 ms.
    #
    # The residual (1.07-1.55x) is not closed here. The known prefill
    # interference factor of 1.18-1.32x accounts for the low-rate end; what
    # remains at 1200 is unexplained and worth a second look.
    duty = min(0.95, max(0.0, q["duty"]))

    # THE QUEUED PREFILL, CHARGED OVER THIS REQUEST'S OWN LIFETIME.
    #
    # `duty` prices the prefill the site was ALREADY doing. It cannot price the
    # prefill the site has not started yet -- the admitted-but-unprefilled
    # backlog, which since the projection fix includes what this scheduler
    # dispatched since the last poll. That work has to run at some point, it
    # blocks decode while it runs, and the request decodes right through it.
    #
    # The horizon is what makes this well-posed, and getting it wrong was the
    # residual. Scored offline over probe_rate5's decision logs, m_tpot/est at
    # p50, charging the backlog over four candidate windows:
    #
    #     window              ca/400   ilp/400   ll/400   rr/400
    #     none (shipped)       1.66      1.69     1.40     1.01
    #     snapshot age         0.25      0.25     1.31     0.96
    #     2000 ms              0.27      0.26     1.36     1.00
    #     1000 ms              0.25      0.25     1.33     0.98
    #     osl x tpot           0.96      1.01     1.35     0.99
    #
    # Only the request's own decode lifetime works, and it is not a fitted
    # choice: the backlog is paid ONCE while this request decodes, so per token
    # it costs work/osl. Every fixed window instead charges it per token as
    # though it recurred, which over-corrects by 4x.
    #
    # No fixed point is needed even though the horizon depends on the answer.
    # With A = 1 - duty and W = work/osl,
    #     t = step / (A - W/t)  =>  A*t - W = step  =>  t = (step + W) / A
    #
    # least_loaded is unmoved (1.40 -> 1.35) because its sites carry no
    # backlog: work_ms is 0 there. Its residual is a different defect.
    if TPOT_FILTER == "batch":
        # THE PROMPT BEING PLACED IS THE POINT. It is this request's own
        # prefill that stops the batch already decoding there, and the arrival
        # test never sees it: req_isl was not even an argument here, while
        # _check_ttft_sla has taken it all along.
        #
        # Not divided by the engine count, unlike everything else in q. The
        # other terms are the average engine's share of a site-wide sum; this
        # prompt lands on ONE engine and stalls that engine's batch for the
        # whole of its prefill. The engines it does not land on are unaffected,
        # and it is the batch it does land on that this test is about.
        backlog = q["prefill_isl"] + q["pending_isl"] + max(0, int(req_isl))
        n_wait = q["f"] + q["p"] + (1 if req_isl > 0 else 0)
    else:
        backlog, n_wait = q["prefill_isl"], q["f"]

    work_ms = 0.0
    if req_osl > 0 and backlog > 0:
        work_ms = analytical_model.ttft_ms(
            total_prefill_tokens=int(backlog),
            bs=max(1, int(n_wait)))
    return (step_ms + work_ms / max(1, req_osl)) / (1.0 - duty)


def estimate_ttft_ms(stats, req_isl: int, analytical_model,
                     fwd_ms: float = 0.0, max_num_seqs: int = 0) -> float:
    """User-perceived TTFT for placing a request at this datacentre.

        ttft = fwd + step_boundary + admission_wait + prefill_drain
             + fused_prefill

    where fused_prefill covers the admitted-but-unprefilled set AND this
    request, since the engine prefills them in one pass.

    fwd_ms is the round-trip the caller reads from the RTT matrix, 0 within a
    region. The user pays it in full before the first token arrives.
    """
    q = _per_engine(stats)
    d, p, R = q["d"], q["p"], q["remaining_osl"]
    f, f_isl = q["f"], q["prefill_isl"]

    if d > 0:
        step_ms = analytical_model.tpot_ms(max(1, int(d)), total_kv=int(q["kv"]))
        # A slot frees when one decoding request runs out of output: the batch
        # owes R tokens and retires d per step, so one slot costs R/d^2 steps.
        slot_ms = (R / d) * step_ms / d
        # Whether this request has to wait for one at all. The occupancy that
        # matters is the site's TOTAL demand -- decoding plus prefilling plus
        # whatever is already queued -- because all three claim a slot before
        # a new arrival does.
        #
        # It used to be `d + p`, which omits the prefilling set: those hold
        # slots but were counted nowhere in the gate. Measured over 41449
        # logged decisions against the queue the run actually charged, at
        # max_num_seqs=128 per engine:
        #
        #     gate        fires   recall   precision
        #     d + p       10.6%      25%         77%
        #     d + f        1.0%       3%         90%
        #     d + f + p   30.1%      68%         73%
        #
        # 32% of decisions met a real queue over 100 ms, so d+f+p firing on
        # 30.1% of them is close to calibrated while d+p missed three
        # quarters. `d + f` alone is worse than either: a site is rarely full
        # of prefills, and ignoring the queue ahead throws away most of the
        # signal.
        #
        # max_num_seqs <= 0 means the cap was never configured. Assume the
        # worst rather than reporting no wait -- but note 0 means "no limit"
        # for bs_cap and "limit unknown" here, which is worth untangling.
        at_cap = (max_num_seqs <= 0) or (d + f + p >= max_num_seqs)
        # p requests are queued ahead, and this one still needs a slot of its
        # own: with p = 0 at a full engine the old form returned zero, which
        # is a wait that certainly happens priced at nothing.
        admission_ms = (p + 1.0) * slot_ms if at_cap else 0.0
    else:
        step_ms = 0.0
        admission_ms = 0.0

    if p > 0 and q["pending_isl"] > 0:
        # The engine fuses what fits in one forward pass. Above the fit's
        # second knee the form is linear, so pricing a backlog larger than one
        # pass as a single pass is within 0.5% of the chunked run the engine
        # would perform; only the fixed per-step cost differs.
        prefill_drain_ms = analytical_model.ttft_ms(
            total_prefill_tokens=int(q["pending_isl"]), bs=max(1, int(p)))
    else:
        prefill_drain_ms = 0.0

    # My prefill does NOT happen alone. The engine fuses the admitted-but-
    # unprefilled set into one forward pass, so what I actually wait for is
    # that whole volume with my own prompt in it -- priced as one pass over
    # (their tokens + mine) at the fused batch size.
    #
    # This used to be ttft_ms(req_isl, bs=1), my prompt only, with the
    # admitted set ignored entirely. It could be ignored because nothing
    # reported it: at admission a request leaves total_pending_isl and joins
    # total_active_isl only once prefilled, so its tokens were in no sum, and
    # DcStats carried prefilling_requests as a COUNT while prefill cost
    # depends on TOKENS. total_prefill_isl was added for exactly this term.
    #
    # What it cost: measured over 43315 logged decisions, the estimate sat
    # flat at 78-364 ms from an empty site all the way to 896 requests
    # resident while the truth climbed 2.6 s -> 25 s, and 85.5% of placements
    # were certified inside a 3000 ms budget when 25.8% were.
    fused_prefill_ms = analytical_model.ttft_ms(
        total_prefill_tokens=int(f_isl + req_isl), bs=max(1, int(f) + 1))

    # The KV gate. The terms above price the seats and the compute; neither
    # sees the cache, and on long prompts the cache fills at ~24 sequences an
    # engine while the seat cap is hundreds, so admission_ms is identically
    # zero exactly where prompts queue longest. The site forecasts this wait
    # from its own requests (DcStats.kv_queue_ms). Once past the gate the
    # prompt runs on its own: the work queued ahead of it was admitted, and
    # computed, during that wait. So the time to the first token is the later
    # of the two -- through the compute queue, or through the gate and then
    # one prefill.
    kv_wait_ms = (float(getattr(stats, 'kv_queue_ms', 0.0) or 0.0)
                  + max(0.0, req_isl
                        - float(getattr(stats, 'kv_room_tokens', 0.0) or 0.0))
                  * float(getattr(stats, 'kv_ms_per_token', 0.0) or 0.0))
    if kv_wait_ms > 0.0:
        through_gate_ms = (min(kv_wait_ms, KV_WAIT_CAP_MS)
                           + analytical_model.ttft_ms(
                               total_prefill_tokens=int(req_isl), bs=1))
        return (fwd_ms + step_ms + admission_ms
                + max(prefill_drain_ms + fused_prefill_ms, through_gate_ms))

    return (fwd_ms + step_ms + admission_ms + prefill_drain_ms
            + fused_prefill_ms)

"""One inference engine: a tensor-parallel group with its own step loop.

The bottom of three layers. `meta_node.Node` owns a set of these and
`meta_dc.Datacenter` owns a set of nodes, and neither reaches past the layer
below it. What lives here is everything an engine does on its own: admission,
batching, the step loop, and the energy integral over its own time series.

A `GPU` here is NOT a card. It is a tensor-parallel group of `tp_size` cards
presenting one engine, one batch and one KV cache; at TP=2 a node of eight
cards holds four of them. The class keeps its old name because it is on every
engine id in every log and result file.


Each GPU tracks concurrently active requests.  Per-request latency is
derived from a latency backend (LatencyLookup or LatencyPredictor) that maps
batch_size → TPOT / TTFT.

**Prefill model (non-chunked):**
- TTFT is computed using only the count of requests currently being
  prefilled (not the total active batch).  Prefill time scales linearly
  with input token count via a per-token prefill rate.
- While any prefill is running, ALL decode threads on the same GPU are
  stalled — their wall-clock latency increases by the prefill duration.
- Multiple requests arriving simultaneously are prefilled as a batch.

**Decode model (segment-based):**
When the batch size changes (request arrives or departs), a new "segment"
begins.  Within a segment every active request progresses at the same
TPOT rate.  The decode loop re-checks the batch size every *chunk_size*
tokens to pick up changes from other threads.

Dispatch is **blocking**: the calling thread sleeps for the full simulated
inference duration (prefill + decode) so that gRPC round-trip time equals
the simulated latency.

**Energy model (time-series based):**
Each GPU records a time series of (timestamp, batch_size) transitions
during simulation.  After the experiment, ``compute_energy(sim_start,
sim_end)`` walks the series and integrates:
- Active segments: ``power_w(batch_size) * duration``
- Idle segments (batch_size=0): ``self._idle_power_w * duration``
  (= IDLE_POWER_W_PER_GPU * tp_size, TP-scaled per pair)
The simulation window [sim_start, sim_end] is determined by the bottleneck
(first request arrival to last request completion across all DCs).
"""

import asyncio
import heapq
import os
import time
import logging
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from datacenter.perf_models.base import LatencyBackend
from datacenter.perf_models.model_specs import kv_bytes_per_token, weight_bytes

log = logging.getLogger(__name__)
from datacenter.perf_models.analytical.energy import StepEnergyModel

# All per-pair power constants are TP-scaled at runtime. The lookup /
# analytical power data was calibrated at POWER_FIT_TP physical GPUs per
# pair (TP=2 for all current models — both LLaMA-8B's and Mixtral-8x7B's
# benchmark power.csv files are TP=2). When the simulator runs at a
# different TP (e.g. Mixtral at TP=4), per-pair power is approximated by
# linearly scaling: power_per_pair(tp) ≈ power_per_pair(fit_tp) · tp/fit_tp.
# Linear is a first-order approximation — actual TP scaling has slight
# sub-linear losses from cross-rank communication, but it's much closer to
# truth than no scaling at all.
POWER_FIT_TP = 2

# Per-physical-GPU constants (multiply by tp_size to get per-pair).
# A100 idle draw is ~50-70W; we use 60W per GPU. A100 TDP at full matmul
# saturation is ~400W; we use that for prefill since matmul units saturate
# near TDP regardless of batch size — the decode power_w(bs) curve
# underestimates prefill because it's fit from steady-state decode data.
# 73 W is the measured band: the coldest idle window of every calibration
# campaign lands at 72.3-75.8 W per GPU across all five (model, TP) pairs. Only
# a fallback -- every pair with an energy artifact overrides it below -- so it
# fires for uncalibrated configs such as deepseek_moe_16b, which the previous
# 60.0 left 18% low. Not a hardware floor: see the note in analyze_energy_fit.py.
IDLE_POWER_W_PER_GPU = 73.0
PREFILL_POWER_W_PER_GPU = 400.0


# ── GPU ──────────────────────────────────────────────────────────────────

class _SeatGate:
    """max_num_seqs, waitable from a thread OR from an event loop.

    `threading.Semaphore` can only do the first, and this engine has to serve
    both. The synchronous path is the CALIBRATED one -- `GPU.process`, which
    `validate_simulation.py` and `testbed/phase2_endtoend/replay_sim.py` still
    call from ordinary threads -- and it must keep working unchanged. The gRPC
    handler now awaits instead, so that a request queued for a seat costs a
    coroutine rather than one of the datacentre's 8000 worker threads.

    One counter, two ways to wait on it, and `acquire`/`release` keep exactly
    the signatures `Semaphore` had so that nothing calling them had to change.

    Releases hand the seat DIRECTLY to the next waiter rather than bumping the
    count and hoping it wakes up in time. The alternative -- wake it and let it
    re-acquire -- loses seats: between the wake and the retry an ordinary
    thread can take the seat, the woken waiter re-parks, and the wake is spent
    with nothing to show for it. Hand-off has no such window.
    """

    def __init__(self, value: int) -> None:
        self._cv = threading.Condition()
        self._value = value
        self._waiters = deque()          # (loop, future), FIFO

    # ── the thread side, unchanged in behaviour from threading.Semaphore ──
    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        with self._cv:
            if not blocking:
                if self._value > 0:
                    self._value -= 1
                    return True
                return False
            if not self._cv.wait_for(lambda: self._value > 0, timeout):
                return False
            self._value -= 1
            return True

    def release(self, n: int = 1) -> None:
        with self._cv:
            self._value += n
            handed = 0
            while self._value > 0 and self._waiters:
                loop, fut = self._waiters.popleft()
                self._value -= 1
                handed += 1
                loop.call_soon_threadsafe(self._deliver, fut, self)
            if n - handed > 0:
                self._cv.notify(n - handed)

    # ── the event-loop side ──
    @staticmethod
    def _deliver(fut, gate) -> None:
        # Already cancelled or timed out: the seat was handed to a waiter that
        # is no longer there, so put it back rather than losing it for the run.
        if fut.done():
            gate.release()
        else:
            fut.set_result(None)

    async def acquire_async(self) -> None:
        with self._cv:
            if self._value > 0:
                self._value -= 1
                return
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._waiters.append((loop, fut))
        await fut


class _DoneFlag(threading.Event):
    """The completion signal, waitable from a coroutine as well as a thread.

    Only `set()` is overridden, so the three places the engine loop finishes a
    request (`r.done.set()`) did not have to change -- and the engine loop is
    the part of this file whose timing the latency model was fitted against.
    """

    def __init__(self) -> None:
        super().__init__()
        self._awaits = []
        self._lk = threading.Lock()

    def set(self) -> None:
        super().set()
        with self._lk:
            waiting, self._awaits = self._awaits, []
        for loop, fut in waiting:
            loop.call_soon_threadsafe(self._resolve, fut)

    @staticmethod
    def _resolve(fut) -> None:
        if not fut.done():
            fut.set_result(None)

    async def wait_async(self, timeout=None) -> bool:
        if self.is_set():
            return True
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self._lk:
            # Re-checked under the lock: `set()` takes the same lock, so a
            # completion landing between the check above and this append cannot
            # be missed.
            if self.is_set():
                return True
            self._awaits.append((loop, fut))
        try:
            await asyncio.wait_for(fut, timeout)
            return True
        except asyncio.TimeoutError:
            return False


class _Req:
    """One in-flight request, as the engine loop sees it."""

    __slots__ = ("isl", "osl", "dataset", "t_start", "done",
                 "prefilled", "generated", "ttft_s", "prefill_bs", "t_end",
                 "wait_s", "step_s", "failed", "t_admit", "pred_osl")

    def __init__(self, isl, osl, dataset, t_start):
        self.isl, self.osl, self.dataset = isl, osl, dataset
        self.t_start = t_start
        self.done = _DoneFlag()
        self.prefilled = 0          # prompt tokens through the engine so far
        self.generated = 0
        self.ttft_s = 0.0
        self.prefill_bs = 0
        self.t_end = t_start
        # The engine-side TTFT splits in two: waiting for whatever step was
        # already running to finish, and the request's own prefill step. Only
        # the second is the latency model; the first is the engine loop, and it
        # scales with how long the steps happen to be.
        self.wait_s = 0.0
        self.step_s = 0.0
        # When the engine took the first chunk of this prompt: the end of its
        # queue. Holding a seat is not being served -- behind a full KV cache
        # a prompt can sit in _waiting for seconds, and that is queueing, the
        # same span vLLM reports as time in queue. 0 until it happens.
        self.t_admit = 0.0
        # The trace's predicted output length, 0 if none; see kv_forecast.
        self.pred_osl = 0
        # Set when the engine loop dies under this request rather than
        # finishing it. Without it a crashed engine leaves every waiter blocked
        # on done.wait() forever, which reads as a hung run rather than a bug.
        self.failed = None


class GPU:
    """A virtual GPU pair (TP group) that supports concurrent batched requests.

    Each GPU object represents a tensor-parallel group (e.g. 2 physical
    GPUs with TP=2).  The benchmark data already reflects TP behaviour,
    so latency lookups are correct as-is.

    **Prefill model (chunked batching):**
    - Prefill requests are queued and batched into chunks of up to
      the engine's prefill token budget, max(max_model_len, 2048).
    - Each chunk is processed sequentially — the GPU computes
      ttft(chunk_total_tokens) for each chunk.
    - Requests whose ISL fits entirely within a chunk finish prefill
      after that chunk. Requests split across chunks finish after
      their last chunk.
    - While any prefill chunk is running, ALL decode threads are
      stalled — their wall-clock latency increases by the chunk duration.

    **Decode model (segment-based):**
    When the batch size changes (request arrives or departs), a new
    "segment" begins. Within a segment every active request progresses
    at the same TPOT rate. The decode loop re-checks the batch size
    every *chunk_size* tokens to pick up changes from other threads.
    """

    # Retired with the step loop. It existed because each request slept its own
    # decode time, so the model had to choose how many tokens to sleep for at a
    # stretch; the answer was one, and the reasons were all artefacts of that
    # design -- energy came out 4.3% high at a chunk of ten and 10.2% at four,
    # non-monotone, an aliasing effect between the chunk period and the arrival
    # pattern. A loop that runs one step and advances every decoding request by
    # one token has no such choice to make. Kept at 1 so anything still reading
    # it sees the only value that was ever correct.
    DECODE_CHUNK = 1

    #: Decode steps to run after a prefill before another prefill may pre-empt
    #: them. 0 -- the default -- is the behaviour that shipped: prefill wins
    #: whenever anything is queued. This is the lever DECODE_CHUNK used to
    #: provide before the step loop, and the one vLLM V0 called
    #: --num-scheduler-steps; see the note in _engine_loop. Set from the
    #: environment so it can be swept without a code change.
    DECODE_STEPS_PER_PREFILL = int(
        os.environ.get("DECODE_STEPS_PER_PREFILL", "0") or 0)

    # A diagnostic, not a setting. Multiplies every step the engine loop sleeps
    # for, prefill and decode alike, so the sensitivity of a result to the step
    # models' own error can be measured rather than argued. The knee is where
    # this matters: batch is arrival rate times residence, residence is the sum
    # of the steps, and the cap is a hard ceiling, so a few percent on the steps
    # moves a disproportionate share of the batch distribution across it. Ship
    # nothing with this at anything but 1.0 -- it is a scale factor fitted to
    # whatever it is pointed at.
    STEP_SCALE = 1.0

    # Per-DECODE-step cost the latency models do not contain, in milliseconds.
    #
    # The models were fitted to engine.step() during a drain: fixed batch, no
    # HTTP, no streaming, nothing else in the process. A serving engine runs
    # its scheduler every step and detokenises and packages a response for
    # every stream in the batch, in the same Python process under the GIL.
    # None of that reached the calibration.
    #
    # Decode only, and absolute rather than a fraction of the step. Both follow
    # from what the work actually is. A decode step detokenises one token for
    # every request in the batch, so the cost scales with the batch; a prefill
    # step emits a first token only for the few prompts it fused, so it carries
    # almost none of it. Charging a fraction of the step got this backwards --
    # a prefill step is long, so it collected twice the absolute overhead of a
    # decode step that was doing thirty times the detokenising.
    #
    # The first fit read the shortfall straight off the TPOT bias -- 0.47,
    # 0.24, 2.18, 3.38 and 2.80 ms at batches 1.3 to 60 -- and doubled the
    # error, because adding time to a step keeps requests in flight longer,
    # which grows the batch, which lengthens the step the model itself prices.
    # The loop roughly doubles whatever is injected. These are those numbers
    # halved, and the halving is what the measurements say:
    #
    #   arm      TPOT bias                  TTFT bias (p50)
    #            none    full    half       none    full    half
    #   r=20    -6.4%   +5.2%   -1.0%      -5.8%   -6.2%   -6.8%  (-7%)
    #   r=24    -7.6%   +7.1%   -1.1%     -10.3%  +28.0%   -3.8%  (-6%)
    #   r=30    -5.2%   +1.2%   -2.1%     -27.8%  +37.1%   +1.6%  (+1%)
    #   r=34    -6.8%   -0.2%   -3.4%     -17.4%  +18.6%   +0.7%  (-1%)
    #
    # Half wins on both quantities at every arm, and at 24 req/s it is the
    # difference between a p90 TTFT of +128% and one of 0%.
    #
    # What it costs is the ceiling: the simulator completes 2.4-2.9% more
    # requests per second than the hardware, where the full term matched it to
    # 0.3%. That is the trade and it is the right way round -- a 2.5% capacity
    # error is easy to state as an envelope, and its direction is known
    # (optimistic), while a 28-37% latency error is not something to report at
    # all.
    #
    # Note this is FITTED to the validation set, unlike the latency and energy
    # coefficients, which come from independent calibration campaigns. Say so
    # when reporting: two parameters against six arms.
    DECODE_OVERHEAD_MS = 0.16
    DECODE_OVERHEAD_MS_PER_REQ = 0.026
    # Tokens in one prefill forward pass, the simulator's stand-in for the
    # engine's `max_num_batched_tokens`. 32768, not 64000, because that is
    # where the energy calibration stops: every campaign swept prefill up to
    # 32768 tokens per step and no further, so a larger pass would ask the
    # model to extrapolate past its measured envelope. It is also a realistic
    # setting -- engines are configured in this range -- so nothing the engine
    # would plausibly do is being clamped away. Longer prompts are split
    # across passes rather than truncated; see _run_prefill_chunks.
    # vLLM with chunked prefill off sets max_num_batched_tokens to
    # max(max_model_len, 2048), and THAT budget -- not a chunking limit -- is
    # what bounds how many queued prompts fuse into one prefill step. The
    # simulator carried a flat 32768, roughly 39 prompts of 839 tokens against
    # the engine's 2.4, which priced its prefill too cheaply. Matching the rule
    # moved the simulated ceiling from 1.5-2.2% optimistic to 0.3-0.5% and the
    # energy error from -2.5% to -1.0% on the TP=2 sweep. It costs TTFT above
    # rho ~ 1, where the metric is dominated by transient onset anyway.
    #
    # Derived rather than constant: 2048 is right for a testbed configured that
    # way, and burning it in would misprice any deployment with a longer
    # context. Set MAX_PREFILL_TOKENS explicitly to override the rule.
    #
    # These two are the DEFAULT, not the setting. 2048 is the testbed's
    # max_model_len and every within-node replay must keep scoring against it,
    # so it stays the fallback; a deployment with a longer context passes
    # max_model_len to the constructor (MAX_MODEL_LEN in the DC's environment)
    # and the budget follows. The paper's workload is 45% above 2048 with a p90
    # of 12190 tokens, so leaving it at the testbed value would chunk a single
    # forward pass into six and charge the 19.3 ms fixed cost of a prefill step
    # six times over -- about 30% of extra TTFT that the engine does not spend.
    # The fitted envelope stops at 32768 tokens in one pass; do not exceed it.
    MAX_MODEL_LEN = 2048
    MAX_PREFILL_TOKENS = 0

    #: Window `prefill_duty` averages over, in seconds. Longer than the load
    #: oscillation it must average across -- measured at about 6 s, which is the
    #: decode residency -- and no longer than the request lifetime the number is
    #: used to predict. Too short and the gap between prefill waves reads as an
    #: idle site; too long and it is the cumulative ratio again.
    DUTY_WINDOW_S = float(os.environ.get("DUTY_WINDOW_S", "10") or 10)
    #: Where kv_forecast reads each request's output length -- the true one,
    #: the trace's prediction, or REF_OSL for every request. The scheduler's
    #: knob, with the scheduler's meaning, so both price the same lengths.
    OSL_SOURCE = os.environ.get("OSL_SOURCE", "oracle")
    REF_OSL = int(os.environ.get("REF_OSL", "500") or 500)
    if OSL_SOURCE not in ("oracle", "predicted", "ref"):
        raise ValueError(f"OSL_SOURCE={OSL_SOURCE!r}; expected oracle, "
                         "predicted or ref")
    #: Most request finishes kv_forecast winds through for the slope.
    KV_FORECAST_MAX_EVENTS = 128

    # KV cache capacity, in tokens. 0 derives it the way vLLM does: profile the
    # device, give the weights what they need and the rest to the cache.
    #
    #     capacity = (tp x HBM x GPU_MEMORY_UTILIZATION - weight_bytes)
    #                / kv_bytes_per_token
    #
    # Until this existed the engine admitted on max_num_seqs and the prefill
    # token budget only, and a prompt was never refused for want of cache --
    # vLLM checks free blocks before it schedules a waiting sequence, and
    # preempts when a running one cannot grow. Derived capacities on the
    # testbed's A100-40GBs: llama3_8B 172k tokens at TP=1 and 467k at TP=2,
    # qwen2.5_7B 408k / 1082k (four KV heads against llama's eight, so half the
    # bytes per token), mixtral_8x7b 467k at TP=4. The paper's trace runs a mean
    # of ~153k tokens in flight, so the gate is slack at TP=2 and NEARLY BOUND
    # at TP=1 -- 89% of capacity at the mean, over it at the peaks.
    HBM_BYTES_PER_GPU = 40 * (1024 ** 3)      # A100-40GB
    GPU_MEMORY_UTILIZATION = 0.90             # vLLM's default; see KV_HEADROOM
    KV_CAPACITY_TOKENS = 0

    # What the cache is allowed to hold, as a fraction of the memory left after
    # the weights:
    #
    #     capacity = KV_HEADROOM * (tp x HBM - weight_bytes) / kv_bytes_per_token
    #
    # NOT vLLM's gpu_memory_utilization, which scales TOTAL device memory
    # before subtracting the weights. This scales what remains, and it is a
    # deliberate idealisation: 0.95 leaves a 5% margin and nothing else, where
    # a real vLLM profiles a forward pass first and reserves whatever
    # activations, CUDA graph buffers and fragmentation need. That reservation
    # is not derivable from the model's geometry and is large -- 21 GiB on
    # Mixtral at TP=4 -- so this runs a MORE generous cache than the hardware
    # provides, by 15-59% depending on the pair. Stated rather than hidden:
    # the simulator is modelling a device that gives the cache everything the
    # weights do not need.
    #
    # The consequence to keep in view is downstream, not here. A larger cache
    # widens the KV region the engine can reach, and the decode fit only
    # covers part of it -- llama3_8B at TP=2 is fitted to 331,200 of the
    # 506,190 this allows, so 65% -- with the rest extrapolated from the
    # boundary gradient. See perf_models/analytical/latency.py.
    KV_HEADROOM = 0.95

    # MEASURED capacities, read off vLLM's own "# GPU blocks" at start-up and
    # recorded as `kv_capacity_tokens` by every calibration campaign. The engine
    # uses the larger KV_HEADROOM derivation unless KV_CAPACITY_SOURCE=measured.
    # These are what the hardware actually gave; the derivation sits 25-93%
    # above them. (The "derived" column used to be the x0.9-utilisation form
    # and read +15-59%; corrected 2026-09-13 to what _derive_kv_capacity
    # actually returns.)
    #
    #   pair                derived    measured
    #   llama3_8B  TP=1     194,894     138,720   +40%
    #   llama3_8B  TP=2     506,190     404,784   +25%
    #   qwen2.5_7B TP=1     459,057     309,152   +48%
    #   qwen2.5_7B TP=2   1,170,591     927,920   +26%
    #   mixtral    TP=4     568,227     293,984   +93%
    #
    # The formula subtracts weights from (devices x HBM x utilisation) and stops
    # there. vLLM profiles a forward pass first and reserves what activations,
    # CUDA graph buffers and fragmentation take -- 21 GiB of it on Mixtral at
    # TP=4. That is not derivable from the model's geometry, only measured.
    #
    # These are at max_model_len 32768, the sweep's setting. The same pair at a
    # different context length profiles differently: llama at TP=2 was measured
    # between 362,672 and 438,832 across campaigns, a 21% spread. Re-read the
    # number from a campaign meta.json rather than assuming it transfers.
    KV_CAPACITY_MEASURED = {
        ("llama3_8B", 1): 138_720,
        ("llama3_8B", 2): 404_784,
        ("qwen2.5_7B", 1): 309_152,
        ("qwen2.5_7B", 2): 927_920,
        ("mixtral_8x7b", 4): 293_984,
    }

    # Class-level so the warning below fires once per process, not once per
    # engine -- a datacentre has eight of them and they all trip together.
    _tail_warned = False
    _clamp_warned = False
    _kv_warned = False

    # Upper bound on one request, as a liveness guard rather than an SLA.
    # Generous: the longest legitimate service seen on the paper's trace is
    # a few tens of seconds, and a replay runs in real time.
    REQUEST_TIMEOUT_S = 900.0

    def __init__(self, gpu_id: str, lookup: LatencyBackend,
                 max_batch_size: int = 0,
                 analytical_model=None,
                 latency_mode: str = "lookup",
                 tp_size: int = POWER_FIT_TP,
                 host_power_share_w: float = 0.0,
                 model_name: str = "",
                 max_model_len: int = 0) -> None:
        self.gpu_id = gpu_id
        self.lookup = lookup
        self.analytical_model = analytical_model  # AnalyticalLatencyModel or None
        self.latency_mode = latency_mode  # "lookup" or "analytical"
        self.max_batch_size = max_batch_size

        # KV is counted in TOKENS by the engine, but energy, cache capacity
        # and transfer cost all scale with BYTES, and bytes per token is an
        # architecture property (GQA decouples it from parameter count --
        # see datacenter/model_specs.py). None for an unknown model: the
        # token count still works, the byte count is simply unavailable
        # rather than silently wrong.
        self.model_name = model_name
        self.max_model_len = max_model_len or self.MAX_MODEL_LEN
        self.max_prefill_tokens = (self.MAX_PREFILL_TOKENS
                                   or max(self.max_model_len, 2048))
        self._kv_bytes_per_token = kv_bytes_per_token(model_name) if model_name else None
        self.kv_capacity_tokens = self._derive_kv_capacity(model_name, tp_size)
        # How many times the cache, rather than the sequence cap, was what kept
        # a queued prompt waiting. Zero means this gate never bound and the run
        # can be read as though it were not there.
        self._kv_blocked_steps = 0
        self._kv_forced = 0
        self._flat_power_warned = False

        # Measured per-step energy model for exactly this (model, TP). None
        # when that pair was never calibrated -- compute_energy() then falls
        # back to the old three-state constants and says so, rather than
        # rescaling a neighbouring TP's coefficients, which was measured to be
        # wrong (per-GPU TP=2/TP=1 ratios 0.736 / 0.713 / 0.366 / 0.628).
        self.energy_model = StepEnergyModel.load(model_name, tp_size)

        # Power scaling — see POWER_FIT_TP comment at module top.
        # tp_scale=1 when tp_size==fit_tp (LLaMA, DeepSeek at TP=2);
        # tp_scale=2 when tp_size=4 / fit_tp=2 (Mixtral).
        self.tp_size = tp_size
        self._tp_scale = float(tp_size) / float(POWER_FIT_TP)
        self._idle_power_w = IDLE_POWER_W_PER_GPU * tp_size
        if self.energy_model is not None:
            # Prefer the measured idle draw over the 60 W/GPU assumption. The
            # one campaign that recorded idle windows read 81.6 W per A100 with
            # weights resident -- 36% above the constant the released simulator
            # used. Idle is charged for every second a DC holds no work, so the
            # error lands squarely on the low-load end of the sweep. Must come
            # after the constant above, which would otherwise overwrite it.
            self._idle_power_w = self.energy_model.idle_w
        self._prefill_power_w = PREFILL_POWER_W_PER_GPU * tp_size

        # This pair's share of its node's host power (CPU/RAM/NIC/fans).
        # Equals HOST_POWER_W / num_pairs_per_node; populated by the Node.
        # Drawn constantly across the full sim window — host stays on
        # whether the GPUs are idle or saturated.
        self._host_power_share_w = host_power_share_w

        self._active_count: int = 0
        # Requests actually DECODING, i.e. past their prefill. _active_count
        # includes requests that have joined but are still waiting for their
        # prompt to be processed, and those contribute no e_d and emit no
        # tokens. Using the wrong one would both over-price the step and
        # mis-divide the step count.
        self._decoding_count: int = 0
        # Wall time lost waiting for decode threads to quiesce before a
        # prefill, and how many prefill chunks paid it. Not a property of the
        # hardware -- an artefact of one-thread-per-request -- so it is
        # measured rather than assumed away.
        # Kept at zero: the step loop needs no barrier, because one loop runs
        # one step and prefill/decode exclusivity is structural. Retained so
        # the replay's reporting still works and the number it prints is the
        # honest one -- there is no longer anything to wait for.
        self._prefill_barrier_s: float = 0.0
        self._prefill_barrier_n: int = 0
        # Wall time actually spent inside a prefill forward pass, with decode
        # blocked. Distinct from "the prefill queue is non-empty", which also
        # counts the barrier -- and during the barrier decode is still running,
        # which is the whole point of waiting for it. Only this is comparable
        # with the engine's prompt throughput divided by its prefill rate.
        self._prefill_busy_s: float = 0.0
        self._t_origin: float = time.monotonic()
        # (t, cumulative busy) after each prefill pass, kept back to the window
        # edge so `prefill_duty` can difference across it. Cumulative-since-boot
        # was what this reported before, and it is diluted by everything that
        # came earlier -- most of all the ramp, when the engine was near idle.
        # Measured on carbon_aware:0.0 at 1200 req/s: the cumulative ratio reads
        # 0.56-0.58 where the plateau itself is 0.65-0.68, so every consumer
        # dividing by (1 - duty) was 20-30% optimistic, the SLA filter included.
        self._duty_samples: Deque[Tuple[float, float]] = deque()
        self._prefill_tokens: int = 0
        # Owned by the engine loop. _waiting holds admitted requests whose
        # prompt is not through yet, _running those that are decoding.
        self._waiting: List["_Req"] = []
        self._running: List["_Req"] = []
        self._engine_on: bool = False
        # What STEP_OVERHEAD added, so a run can report it rather than leave it
        # buried in the step times.
        self._overhead_s: float = 0.0
        self._step_count: int = 0
        # Energy drawn during those overhead windows. The decode energy path
        # counts steps and never looks at elapsed time, which is right for the
        # step itself and wrong for the window DECODE_OVERHEAD_MS adds after
        # it: the latency model says the step now takes longer, and the energy
        # model has no way to know. Accounted here, where both the step's
        # duration and its overhead are in hand.
        self._overhead_energy_j: float = 0.0
        # Monotonic count of tokens this engine has ever emitted. Distinct
        # from _total_generated_tokens, which is the live figure and drops
        # when a request leaves. A difference of THIS between two marks is
        # the tokens decode produced in between, and dividing it by the
        # decoding count gives the step count exactly -- with no need to
        # record a mark on every decode chunk.
        self._cum_generated: int = 0
        # Prompt tokens of every request ever submitted; with _total_requests,
        # the mean prompt kv_forecast prices a new arrival at.
        self._cum_isl: int = 0
        self._pending_count: int = 0  # requests waiting on semaphore
        self._prefill_count: int = 0
        # Prompt tokens of requests that have been ADMITTED and are waiting for
        # their prefill step. Reported because nothing else covers them: at
        # admission the request leaves _total_pending_isl, and it does not join
        # _total_active_isl until its prefill lands, so between those two points
        # its tokens appear in no sum at all. That is the window a new arrival
        # fuses its own prefill into, and prefill cost is a function of tokens,
        # so a scheduler with only the COUNT cannot price it. Measured: the
        # TTFT estimator, blind to this, certified 85.5% of placements as
        # inside a 3000 ms budget when 25.8% were.
        self._total_prefill_isl: int = 0
        self._total_pending_isl: int = 0  # sum of ISLs of requests waiting on semaphore
        self._total_active_isl: int = 0  # sum of ISLs of active (decoding) requests
        # Sum of tokens generated so far across all currently-active (decoding)
        # requests. total_KV fed to the per-step lookup is
        # _total_active_isl + _total_generated_tokens.
        self._total_generated_tokens: int = 0
        # Decode-step work the current batch still owes: sum of (osl -
        # generated) over the decoding set. Maintained incrementally beside
        # _total_generated_tokens so the two always describe the same
        # population. A request contributes osl when it joins the decode loop
        # and reaches 0 exactly when it leaves, so nothing has to be
        # subtracted on completion.
        self._total_remaining_osl: int = 0
        self._lock = threading.Lock()
        self._total_requests: int = 0

        # Admission control: blocks callers when GPU pair is at capacity
        self._semaphore = (_SeatGate(max_batch_size)
                           if max_batch_size > 0 else None)

        # Prefill queue: requests wait here for their prefill to complete.
        # Each entry is (in_tok, done_event) — the request thread waits
        # on done_event until the prefill scheduler processes it.
        self._prefill_queue: List[Tuple[int, threading.Event]] = []
        self._prefill_running = False  # True while a prefill batch is being processed

        # Event cleared during prefill → decode threads block on .wait()
        # Decode steps currently executing. A real engine schedules a prefill
        # at a STEP BOUNDARY: whatever step is running finishes, and the next
        # one is the prefill -- the two never share a forward pass under
        # V0 with chunked prefill off, which is the regime every campaign
        # measured. Clearing the event alone does not reproduce that, because
        # a thread that has already begun its sleep runs straight through the
        # prefill. Prefill therefore waits for this to reach zero before it
        # starts, which costs it at most one step time.

        # Time series of (monotonic_time, batch_size, kv_tokens) at every
        # discontinuity. Used by compute_energy() after the experiment.
        #
        # Entries are recorded where a quantity JUMPS: a request joining or
        # leaving the batch, and a request's prompt entering the cache when
        # its prefill finishes. Between two entries the batch size is
        # constant and the cache grows one token per decoding request per
        # step, i.e. linearly, so the segment is integrated in closed form
        # rather than sampled. Appending on every decode chunk would make the
        # series enormous for no accuracy the linear model does not already
        # give.
        self._time_series: List[Tuple[float, int, int, int, int]] = []
        # Per-chunk prefill events (start_time, end_time). Each entry marks
        # a segment during which the GPU was doing prefill compute at
        # PREFILL_POWER_W, with the decode loop paused. Used by
        # compute_energy() to correctly attribute prefill energy separately
        # from decode / idle energy.
        # (t0, t1, n_tok, n_reqs, sum_sq_isl). The last is sum_i(isl_i^2),
        # which the attention term is fitted against -- exact here because
        # the step knows its own sequences, so no equal-length assumption.
        self._prefill_events: List[Tuple[float, float, int, int, float]] = []
        self._prefill_state: List[Tuple[int, int, int]] = []
        self._oversleep_s: float = 0.0
        self._asked_s: float = 0.0
        # Energy (joules) — populated by compute_energy() after experiment.
        self._energy_j: float = 0.0
        # The host share alone, so a caller can compare the GPU part against a
        # power meter. nvidia-smi and NVML see the devices, not the node.
        self._host_energy_j: float = 0.0
        # Energy this engine drew doing nothing: powered, weights resident, no
        # request in flight. Split out rather than buried in the total because
        # including it is a modelling CHOICE, not a fact. If a site would have
        # been powered anyway, idle is sunk and a policy comparison should be
        # made on the work; if the fleet is provisioned for this workload, idle
        # is a real cost of owning it. Both are defensible and a paper should
        # be able to show either from one run, which a flag would not allow --
        # it would produce two result sets that cannot be put on one axis.
        # Only the measured path fills this; the legacy three-state path prices
        # idle inside its interval loop and leaves this at 0.
        self._idle_energy_j: float = 0.0
        # Requests whose output was capped at the context window; see process().
        self._osl_clamped: int = 0
        # Set by the engine-loop crash handler. `_engine_on` is NOT a health
        # flag -- the loop clears it whenever it goes idle, so it reads False
        # on a perfectly healthy engine. This is the one that means something.
        self._engine_faults: int = 0
        self._last_fault: str = ""
        # Batch-size profile: cumulative seconds at each bs (set by compute_energy)
        self._time_at_bs: Dict[int, float] = defaultdict(float)
        # Cumulative prefill seconds — set by compute_energy() for diagnostics.
        self._prefill_seconds: float = 0.0
        # Set by the measured-model path: how many decode steps were priced,
        # and how long the engine sat idle. Both are diagnostics -- a run whose
        # step count does not match its generated-token count has a bug.
        self._decode_steps: float = 0.0
        self._decode_steps_per_prefill = self.DECODE_STEPS_PER_PREFILL
        #: Decode steps still owed before prefill may run again. Raised when a
        #: prefill attempt is blocked by the cache, not after every prefill.
        self._decode_debt = 0
        # Tokens the priced decode steps account for. Must equal the
        # tokens the run actually generated -- if it does not, an
        # interval was skipped and its energy is missing.
        self._decode_tokens: int = 0
        self._idle_seconds: float = 0.0

    def _derive_kv_capacity(self, model_name: str, tp_size: int):
        """Tokens the cache holds, or None when the model geometry is unknown.

        None disables the gate rather than assuming a capacity: a model with no
        weight_bytes entry would otherwise silently get an infinite cache, which
        is the failure mode this check exists to remove.
        """
        if self.KV_CAPACITY_TOKENS:
            return int(self.KV_CAPACITY_TOKENS)
        # KV_CAPACITY_SOURCE=measured takes vLLM's own profiled capacity for this
        # (model, TP) instead of the idealisation below. It decides the answer
        # once the cache binds: long prompts fill Mixtral TP=4 at ~26 sequences
        # on the measured 293,984 tokens but ~51 on the derived 568,227, twice
        # the KV range the decode fit was calibrated on. A pair with no
        # measurement refuses to start rather than mix two sources in one run.
        if os.environ.get("KV_CAPACITY_SOURCE", "derived").strip().lower() == "measured":
            measured = self.KV_CAPACITY_MEASURED.get((model_name, tp_size))
            if not measured:
                raise ValueError(
                    f"KV_CAPACITY_SOURCE=measured but no profiled capacity is "
                    f"recorded for ({model_name!r}, TP={tp_size}); add it to "
                    f"KV_CAPACITY_MEASURED or run with the derived capacity")
            return int(measured)
        per_token = self._kv_bytes_per_token
        w = weight_bytes(model_name) if model_name else None
        if not per_token or not w:
            return None
        # Everything the weights leave behind, less a 5% margin. See
        # KV_HEADROOM: this is the idealisation, and it is deliberately more
        # generous than the profiled capacities in KV_CAPACITY_MEASURED, which
        # override it only under KV_CAPACITY_SOURCE=measured. Those were read off
        # vLLM's own "# GPU blocks" and come in 20-48% lower because a real
        # engine reserves memory this does not model.
        free = tp_size * self.HBM_BYTES_PER_GPU - w
        return int(free * self.KV_HEADROOM / per_token) if free > 0 else None

    def _mark(self) -> None:
        """Record a time-series point. Caller must hold self._lock.

        Five fields because the energy model needs to tell two kinds of cache
        growth apart. `generated` rises only when a decode step runs, by
        exactly one token per decoding request, so a difference in it divided
        by the decoding count IS the number of steps. `kv_tokens` also jumps
        when a prompt lands in the cache at the end of a prefill, which is not
        a decode step and must not be counted as one.
        """
        self._time_series.append((
            time.monotonic(),
            self._active_count,
            self._decoding_count,
            self._total_active_isl + self._total_generated_tokens,
            self._cum_generated,
        ))

    @property
    def kv_tokens(self) -> int:
        """Tokens currently held in the KV cache across active requests."""
        return self._total_active_isl + self._total_generated_tokens

    def _forecast_osl(self, r) -> int:
        """The output length kv_forecast assumes for r, per OSL_SOURCE."""
        if self.OSL_SOURCE == "oracle":
            return r.osl
        if self.OSL_SOURCE == "predicted" and r.pred_osl > 0:
            return r.pred_osl
        return self.REF_OSL

    def _forecast_prefill_s(self, isl: float) -> float:
        """One prompt's prefill, in the chunks the engine would run it in."""
        t, left = 0.0, int(isl)
        while left > 0:
            c = min(left, self.max_prefill_tokens)
            t += self.STEP_SCALE * self._get_ttft_chunk([(c, None, "")])
            left -= c
        return t

    def kv_forecast(self) -> Tuple[float, float, float, float]:
        """When the KV gate lets the queue in, wound forward from now.

        Three stages, on this engine's own requests:

          1. the running ones finish in order of their remaining output, each
             handing back its prompt and output, while every decode step grows
             the cache by one token per running request;
          2. the prompts waiting FCFS are let in as soon as the room covers the
             head one -- each pays its prefill, during which decode stalls, and
             then decodes its own output;
          3. once the queue is in, the same wind continues until a new prompt
             of the mean length seen here would fit.

        Returns (ahead_ms, room_tokens, ms_per_token, ref_isl). ahead_ms is the
        time stages 1-2 take, and 0 when the gate never held the queue: then
        the wait ahead is compute, which the scheduler prices on its own.
        room_tokens is the cache left once the queue is in. A new prompt of y
        tokens waits

            ahead_ms + max(0, y - room_tokens) * ms_per_token

        where ms_per_token is stage 3's secant, through a prompt ref_isl
        tokens past the room. Output lengths come from OSL_SOURCE; a prompt
        still on a seat is priced at the mean of those on seats and REF_OSL.
        """
        cap = self.kv_capacity_tokens
        if not cap:
            return 0.0, 0.0, 0.0, 0.0
        with self._lock:
            running = []
            for r in self._running:
                o = max(self._forecast_osl(r), r.generated + 1)
                running.append((o - r.generated, r.isl + o))
            waiting = [(r.isl, r.prefilled, self._forecast_osl(r))
                       for r in self._waiting]
            n_seat, isl_seat = self._pending_count, self._total_pending_isl
            used = self._total_active_isl + self._total_generated_tokens
            ref = self._cum_isl / self._total_requests if self._total_requests else 0.0

        free = float(cap - used)
        t, k = 0.0, 0
        heap = [(rem, frees) for rem, frees in running]
        heapq.heapify(heap)
        queue = deque()
        for isl, done, osl in waiting:
            if done:
                # Mid-prefill: past the gate already, but only the chunks run
                # so far are in `used`; the rest of its prompt is committed.
                free -= isl
                t += self._forecast_prefill_s(isl - done)
                heapq.heappush(heap, (k + max(osl - 1, 1), isl + max(osl, 1)))
            else:
                queue.append((isl, osl))
        for _ in range(n_seat):
            queue.append((isl_seat / n_seat, self.REF_OSL))

        def advance():
            nonlocal free, t, k
            kf = heap[0][0]
            n = len(heap)
            if kf > k:
                kv = cap - free + n * (kf - k) / 2.0
                step = (self.STEP_SCALE * self._get_tpot(n, total_kv=int(kv),
                                                         avg_isl=kv / n)
                        + (self.DECODE_OVERHEAD_MS
                           + self.DECODE_OVERHEAD_MS_PER_REQ * n) / 1000.0)
                t += (kf - k) * step
                free -= n * (kf - k)
                k = kf
            while heap and heap[0][0] == k:
                free += heapq.heappop(heap)[1]

        blocked = False
        while queue:
            isl, osl = queue[0]
            # The engine admits anyway when nothing is decoding; so does this.
            if free >= isl or not heap:
                queue.popleft()
                free -= isl
                t += self._forecast_prefill_s(isl)
                heapq.heappush(heap, (k + max(osl - 1, 1), isl + max(osl, 1)))
            else:
                blocked = True
                advance()
        t_in, room = t, free

        need = max(room, 0.0) + max(ref, 1.0)
        events = 0
        while free < need and heap and events < self.KV_FORECAST_MAX_EVENTS:
            advance()
            events += 1
        gained = (need if free >= need or not heap else free) - room
        ms_per_token = 1000.0 * (t - t_in) / max(gained, 1.0)
        return (1000.0 * t_in if blocked else 0.0), room, ms_per_token, ref

    @property
    def load(self) -> int:
        """Requests in flight. The metric `least_loaded` keys on.

        Named the same on Node, so one policy class works at both levels
        without knowing which it is placing on.
        """
        return self._active_count

    def _get_kv_capacity_tokens(self) -> int:
        """Tokens this engine's KV cache can hold, 0 if unmeasured.

        Measured from the engine at calibration time, not derived: deriving it
        as tp x HBM x utilisation - weights overstates what vLLM actually
        reports by 15-59%.
        """
        return int(self.kv_capacity_tokens or 0)

    @property
    def kv_utilisation(self) -> float:
        """Share of this engine's KV cache in use, 0.0 if capacity is unknown.

        The metric `kv_headroom` keys on. It is not the same ordering as
        `load`: two engines can hold the same number of requests and very
        different amounts of cache, because a long prompt occupies a seat and
        a large multiple of the memory a short one does. Admission is bounded
        by BOTH -- max_num_seqs and free blocks -- and only the first of them
        has ever bound on this testbed.
        """
        cap = self._get_kv_capacity_tokens()
        return (self.kv_tokens / cap) if cap > 0 else 0.0

    @property
    def kv_bytes(self) -> int:
        """Bytes those tokens occupy, or 0 if the model geometry is unknown."""
        b = self._kv_bytes_per_token
        return 0 if b is None else b * self.kv_tokens

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def pending_count(self) -> int:
        return self._pending_count

    @property
    def decoding_count(self) -> int:
        """Requests past their prefill -- the batch the next step actually runs."""
        return self._decoding_count

    @property
    def prefill_count(self) -> int:
        return self._prefill_count

    @property
    def total_prefill_isl(self) -> int:
        """Prompt tokens admitted but not yet prefilled. See __init__."""
        return self._total_prefill_isl

    @property
    def total_pending_isl(self) -> int:
        return self._total_pending_isl

    @property
    def total_active_isl(self) -> int:
        return self._total_active_isl

    @property
    def total_remaining_osl(self) -> int:
        """Decode steps the current batch still owes, summed over decoders.

        Divided by the decoding count this is the mean remaining output
        length, which is what sets how soon a sequence slot frees.
        """
        return self._total_remaining_osl

    @property
    def engine_faults(self) -> int:
        """Times this engine's loop died. Nonzero means it dropped requests."""
        return self._engine_faults

    @property
    def last_fault(self) -> str:
        return self._last_fault

    @property
    def osl_clamped(self) -> int:
        """Requests whose output was cut to fit the context window.

        Computed since the clamp was added and read by nothing until now,
        although the comment beside it claimed the count was reported per run.
        Rare but not harmless: 2 of 2107 requests on mixed_trace_3ds carry
        1.84% of every decode token in it.
        """
        return self._osl_clamped

    @property
    def kv_blocked_steps(self) -> int:
        """Prefill admissions refused for want of KV blocks.

        The gate is measured never to bind on this testbed, where the cache
        runs 15-33% full -- but that was measured, and this is how it stays
        measured instead of assumed.
        """
        return self._kv_blocked_steps

    @property
    def oversleep_s(self) -> float:
        """Wall-clock the HOST cost on top of what the model asked for.

        The replay is real time -- `_engine_step_loop` prices a step and then
        sleeps it -- so a contended host lengthens every step and the fleet
        reads slower for a reason that is not the model. The comment at the
        sleep site records the consequence measured earlier: the same trace
        once scored 5-7 points apart on a laptop and on the server.

        This was accumulated from the start and reported NOWHERE, so the
        contamination has never been quantified at any operating point. Read it
        as `oversleep_s / asked_s`: the fraction by which the host stretched
        the simulated clock. It is the number that decides whether a rate is
        faithfully simulable, and it is load-dependent -- one blocked thread
        per in-flight request means the concentrating policies, which hold the
        most requests in flight, are where it should be read.
        """
        return self._oversleep_s

    @property
    def asked_s(self) -> float:
        """Wall-clock the model asked for, the denominator of the above."""
        return self._asked_s

    @property
    def kv_forced(self) -> int:
        """Prompts admitted despite exceeding the WHOLE KV cache.

        vLLM would preempt or fail here and neither is modelled, so the engine
        proceeds and counts it. The warning at the site says "Count reported
        per run" -- it was not, until this. A non-zero value means the run
        contains behaviour the simulator does not model.
        """
        return self._kv_forced

    @property
    def total_generated_tokens(self) -> int:
        """Sum of tokens generated-so-far across active decoding requests.

        Combined with ``total_active_isl`` this gives the instantaneous
        total_KV that feeds the per-step TPOT lookup:
            total_KV = total_active_isl + total_generated_tokens
        """
        return self._total_generated_tokens

    def _get_ttft_chunk(self, chunk: list) -> float:
        """Get TTFT for a prefill chunk in seconds.

        Args:
            chunk: list of (in_tok, done_event, dataset) tuples.

        Analytical mode: treats the whole chunk as one compute block
            using total tokens (hardware-accurate for fused prefill).
        Lookup mode: prefers the batched (BS, total_prefill_tokens) lookup
            from the 3-dataset drain pool; falls back to summing per-request
            BS=1 TTFTs if that lookup isn't loaded.
        """
        if self.latency_mode == "analytical" and self.analytical_model is not None:
            chunk_tokens = sum(in_tok for in_tok, _, _ in chunk)
            # The exact sum of squares, not the equal-lengths approximation the
            # model falls back to. A step here routinely fuses one long prompt
            # with several short ones -- the paper's trace runs 61-token MBPP
            # prompts beside 12000-token GovReport ones -- and attention is
            # quadratic within a sequence, so which shape the step has changes
            # its cost by ~9% at 12288 total tokens. The engine knows the
            # lengths; there is no reason to guess them.
            sum_sq = sum(in_tok * in_tok for in_tok, _, _ in chunk)
            return self.analytical_model.ttft_s(chunk_tokens, bs=len(chunk),
                                                sum_sq=sum_sq)
        # Lookup mode
        if self.lookup is not None and self.lookup.has_ttft_batch_lookup():
            chunk_bs = len(chunk)
            chunk_tokens = sum(in_tok for in_tok, _, _ in chunk)
            return self.lookup.ttft_batch_s(chunk_bs, chunk_tokens)
        # Fallback: sum of per-request BS=1 TTFT (legacy behavior)
        total = 0.0
        for in_tok, _, dataset in chunk:
            total += self.lookup.ttft_by_isl(in_tok, dataset=dataset)
        return total

    def _get_tpot(self, bs: int, total_kv: int = 0, avg_isl: float = 0) -> float:
        """Get per-engine-step decode time in seconds.

        Analytical mode uses the legacy ``(bs, avg_isl)`` API.
        Lookup mode uses the new binned ``(bs, total_kv)`` lookup when
        the 3-dataset table is loaded; falls back to BS-only TPOT otherwise.
        """
        if self.latency_mode == "analytical" and self.analytical_model is not None:
            # total_kv, not the (bs, avg_isl) legacy pair. avg_isl counts only
            # prompt tokens, so reconstructing the cache from it treats every
            # step as the batch's first -- it drops every token generated so
            # far. The caller already has the exact figure; passing it is the
            # difference between a KV term evaluated at the truth and one
            # evaluated at the start of decode.
            return self.analytical_model.tpot_s(bs, avg_isl, total_kv=total_kv)
        if self.lookup is not None and self.lookup.has_step_ms_lookup():
            return self.lookup.step_s(bs, total_kv)
        return self.lookup.tpot(bs)

    def prefill_duty(self) -> float:
        """Share of RECENT wall time this engine spent with decode blocked.

        Over the last DUTY_WINDOW_S, or since the engine started if that is
        shorter. A cumulative ratio answers a question nobody asks -- what the
        engine has averaged since boot -- and is dragged down by the ramp for
        the whole of a run. What every consumer wants is what the site is doing
        NOW, because that is what the request being placed will live through.

        The window is longer than the load oscillation it has to average over
        (measured at about 6 s, the decode residency) and no longer than the
        request lifetime it is used to predict, so it neither reads the gap
        between prefill waves as idle nor smooths away a site that has genuinely
        filled up.

        Bounded below 1: the caller divides by (1 - duty), and an engine that
        measures 1.0 for a moment would otherwise produce an infinite estimate.
        """
        now = time.monotonic()
        with self._lock:
            busy_now = self._prefill_busy_s
            start = max(self._t_origin, now - self.DUTY_WINDOW_S)
            busy_then = 0.0 if start <= self._t_origin else None
            if busy_then is None:
                # Last sample at or before the window edge; the deque is kept
                # in time order and pruned to hold exactly one such entry.
                busy_then = 0.0
                for t, b in self._duty_samples:
                    if t <= start:
                        busy_then = b
                    else:
                        break
        el = now - start
        if el <= 0:
            return 0.0
        return min(0.95, max(0.0, (busy_now - busy_then) / el))

    def decode_power_w(self, bs: int, kv_tokens: float, step_s: float) -> float:
        """Instantaneous draw of a decode step, in watts.

        Power is a property of the step's composition -- phase, batch size,
        cache occupancy -- and does not depend on elapsed time. Time enters
        only here, dividing the step's energy by its duration. That is also
        the one place a latency error becomes a power error, which the device
        TDP makes visible: a step time that is too short pushes this above the
        ceiling.
        """
        m = self.energy_model
        if m is None or step_s <= 0:
            return self._get_power_w(bs)
        return m.decode_step_j(bs, kv_tokens) / step_s

    def prefill_power_w(self, n_tokens: int, step_s: float) -> float:
        """Instantaneous draw of a prefill step, in watts.

        Not a constant. Above the plateau threshold L* the step does saturate
        the device and a constant TDP is right; below it the draw falls well
        short -- 274 W against 400 for Llama-3.2-3B at 128 tokens in the step
        -- and L* itself ranges from 128 to 1792 tokens depending on the model
        and the parallelism degree.
        """
        m = self.energy_model
        if m is None or step_s <= 0:
            return self._prefill_power_w
        return m.prefill_step_j(n_tokens) / step_s

    def _get_power_w(self, bs: int) -> float:
        """Decode power in watts for a GPU pair, TP-scaled to the runtime tp_size.

        Only reached for a model with no StepEnergyModel; anything calibrated
        integrates the measured per-step model instead and never comes here.
        The analytical log-linear curve was calibrated at POWER_FIT_TP physical
        GPUs per pair, so runtime scaling brings it to the actual tp_size.

        The third branch used to interpolate a per-model power.csv. It is gone:
        the two post-processors that wrote those files disagreed on convention
        -- LLaMA's per GPU, Mixtral's per pair -- and nothing in the CSV said
        which, so the same number meant different things depending on which
        model's file was passed. A flat constant that is obviously a constant
        is safer than a table that is silently in the wrong units.
        """
        if self.latency_mode == "analytical" and self.analytical_model is not None:
            base = self.analytical_model.power_w(bs)
        else:
            if not self._flat_power_warned:
                self._flat_power_warned = True
                log.warning(
                    "%s: no energy model and no analytical curve -- decode "
                    "power is the flat %.0f W placeholder. Energy from this "
                    "engine is not a measurement.", self.gpu_id, 300.0)
            base = 300.0
        return base * self._tp_scale

    # ── the engine loop ───────────────────────────────────────────────────
    #
    # One thread per GPU slot, not one per request. A real engine is a single
    # loop: each iteration decides what the next step is, runs it, and advances
    # everything that step touched. Modelling it that way makes three things
    # structural that the previous design had to arrange by hand.
    #
    # Prefill and decode cannot overlap, because one loop runs one step. The
    # old design had ~60 request threads sleeping independently and needed a
    # barrier -- wait until none of them is mid-step -- to keep the two apart.
    # That barrier cost 0.3% of engine wall time at batch 1.3 and 12.1% at 60,
    # and it grew with load exactly as the TTFT error did.
    #
    # Prefill batches by itself. Work that arrives while a step is running is
    # in the queue when the step ends, so the next prefill step carries all of
    # it and pays the fixed per-step cost once. The old design let the first
    # request to find an empty queue start prefilling alone: 1.05 prompts per
    # step against the hardware's 2.4.
    #
    # And the thread count stops tracking the batch size. At 30 req/s the old
    # design woke ~3300 times a second across three slots; this wakes ~54. The
    # replay was measurably machine-sensitive because of it -- the same trace
    # scored 5 to 7 points differently on a laptop and on the server.

    def _ensure_engine(self) -> None:
        """Start the slot's engine thread if it is not already running.

        Caller must hold self._lock. Started lazily and allowed to exit when
        the slot goes idle, so an unused slot costs nothing.
        """
        if self._engine_on:
            return
        self._engine_on = True
        threading.Thread(target=self._engine_loop, daemon=True).start()

    def _next_prefill_batch(self):
        """Take up to the engine's prefill token budget from the front.

        Caller must hold self._lock. Returns (pieces, reqs_completed) where
        pieces is what the latency model prices and reqs_completed are the
        requests whose prompt finishes in this step.

        A prompt longer than the cap is split across steps, which is what the
        engine does -- max_num_batched_tokens bounds a forward pass -- and also
        keeps the energy model inside the envelope it was fitted on.
        """
        # What the engine saw when it chose this step: how many prompts were
        # queued and how many sequence slots decode had left free. vLLM's cap
        # is on the RUNNING sequences with the queue outside it, so its free
        # count is max_num_seqs - running; this engine's semaphore caps queue
        # plus running together, so its free count is max_num_seqs - decoding
        # and the queued prompts are already inside the cap. Whether those two
        # differ, and by how much, is what decides the fusion width and with it
        # the prefill step time.
        self._prefill_state.append(
            (len(self._waiting), self._decoding_count, self._active_count))
        now = time.monotonic()
        for r in self._waiting:
            if r.wait_s == 0.0 and r.prefilled == 0:
                r.wait_s = now - r.t_start
        pieces, completed, room = [], [], self.max_prefill_tokens
        cap = self.kv_capacity_tokens
        # What the cache already holds, plus what this step is about to add.
        kv_used = self._total_active_isl + self._total_generated_tokens
        while self._waiting and room > 0:
            r = self._waiting[0]
            # The cache gate, checked before the prompt is taken and not after.
            # vLLM asks can_allocate() for a waiting sequence and leaves it in
            # the queue when the blocks are not there; the sequence cap and the
            # token budget cannot express that, so before this the engine would
            # admit a prompt into a cache with no room for it.
            #
            # FCFS: break rather than skip ahead. Letting a short prompt jump a
            # blocked long one would starve the long one and quietly reorder the
            # queue the latency model prices.
            if cap is not None and r.prefilled == 0 and kv_used + r.isl > cap:
                self._kv_blocked_steps += 1
                break
            if r.prefilled == 0:
                r.t_admit = now
            take = min(r.isl - r.prefilled, room)
            pieces.append((take, r.dataset))
            r.prefilled += take
            room -= take
            kv_used += take
            if r.prefilled >= r.isl:
                completed.append(self._waiting.pop(0))
        return pieces, completed

    def _engine_loop(self) -> None:
        """The slot's step loop. Never allowed to die quietly.

        This used to be an uncaught `while True` on a daemon thread. An
        exception inside it -- and there was one, a stale `self.name` in a
        warning path -- killed the thread with `_engine_on` still True, so
        `_ensure_engine` believed an engine was running and started no
        replacement. Every request on that slot then blocked on `done.wait()`
        for the length of the run, its semaphore permit never returned, and the
        only symptom was a datacentre that had quietly stopped answering. The
        wrapper below turns that into a logged failure and a set of requests
        that come back as errors.
        """
        try:
            self._engine_step_loop()
        except BaseException as e:            # noqa: BLE001 - deliberate
            log.exception("%s: engine loop died (%s); failing its requests",
                          self.gpu_id, type(e).__name__)
            with self._lock:
                self._engine_on = False
                self._engine_faults += 1
                self._last_fault = f"{type(e).__name__}: {e}"
                waiting = list(self._waiting)
                running = list(self._running)
                self._waiting.clear()
                self._running.clear()
                # Roll the counters back for everything just dropped. Clearing
                # the queues used to be the whole of it, which left the engine
                # reporting its crash-time load for the rest of the run: eight
                # requests failed out to their callers and the datacentre went
                # on publishing active=8, decoding=8, remaining_osl=24 with
                # nothing left that could ever decrement them. Every consumer
                # reads that -- avg_batch_size, the ILP's dc_bs, _per_engine's
                # d, the provisioner's RTT heat -- so a dead engine did not
                # merely stop serving, it made its whole site look permanently
                # busy and never recovered.
                #
                # A waiting request has been counted active and prefilling and
                # has its prompt in _total_prefill_isl; the decode-side sums
                # are added when its prefill lands.
                # A running one has been counted everywhere, and owes
                # osl - generated of its output.
                for r in waiting:
                    self._active_count -= 1
                    self._prefill_count -= 1
                    self._total_prefill_isl -= r.isl
                for r in running:
                    self._active_count -= 1
                    self._decoding_count -= 1
                    self._total_active_isl -= r.isl
                    self._total_remaining_osl -= max(0, r.osl - r.generated)
                    self._total_generated_tokens -= r.generated
                self._mark()
            for r in waiting + running:
                r.failed = f"{type(e).__name__}: {e}"
                r.t_end = time.monotonic()
                r.done.set()
            raise

    def _engine_step_loop(self) -> None:
        while True:
            with self._lock:
                if not self._waiting and not self._running:
                    self._engine_on = False
                    return
                pieces = completed = None
                # Prefill has absolute priority unless a decode run is owed.
                #
                # The step loop replaced a design in which each request slept
                # its decode time in DECODE_CHUNK-token stretches, and prefill
                # interleaved between those stretches. That knob went; the
                # arbitration it implemented did not get replaced, and what
                # took its place -- prefill whenever anything is queued -- is
                # the harshest setting for decode there is. At high arrival
                # rates `_waiting` never empties, so decode runs only when the
                # queue dries up or the KV gate blocks a prompt. Measured on
                # grid_v3: prefill_duty 0.13-0.55, rising with concentration,
                # and TPOT is charged for all of it because decode is stalled
                # inside the request's own service window.
                #
                # vLLM V0 had the same lever as `--num-scheduler-steps`, and
                # dropped it in V1 in favour of chunked prefill, which puts
                # prefill chunks and decode in ONE step so decode is never
                # stalled at all. Modelling that needs a cost for a mixed step,
                # which was never measured. Reordering steps needs no new cost
                # model -- the steps themselves are unchanged -- so this is the
                # honest lever to expose while V1's is out of reach.
                #
                # 0 keeps today's behaviour exactly.
                # The debt is owed for a BLOCKED prefill, not for every one.
                # Prefill runs freely while the cache has room; when the gate
                # stops it, one decode step rarely frees enough to admit the
                # next prompt, so the engine oscillates -- block, one decode,
                # block. Paying N decodes at once frees a usable amount, and
                # the prefills that fit in it then run back to back.
                if self._waiting and not self._decode_debt:
                    pieces, completed = self._next_prefill_batch()
                    if not pieces:
                        # Blocked on cache. Owe a decode run before trying
                        # again; see the note above.
                        self._decode_debt = self._decode_steps_per_prefill
                        # Every queued prompt is waiting on cache, not on a
                        # slot. Decode instead: it is the only thing that frees
                        # KV, so spinning on an empty prefill step would hang
                        # the engine. With nothing decoding either there is
                        # nothing to wait for, and the engine admits one anyway
                        # -- vLLM would preempt or OOM here, and neither is
                        # modelled, so the honest thing is to proceed and count
                        # it rather than deadlock silently.
                        if not self._running:
                            self._kv_forced += 1
                            if not GPU._kv_warned:
                                GPU._kv_warned = True
                                log.warning(
                                    "%s: a prompt exceeds the whole KV cache "
                                    "(%s tokens); admitting it anyway. vLLM "
                                    "would preempt or fail, which this "
                                    "simulator does not model. Count reported "
                                    "per run.", self.gpu_id,
                                    self.kv_capacity_tokens)
                            r = self._waiting[0]
                            if r.prefilled == 0:
                                r.t_admit = time.monotonic()
                            take = min(r.isl - r.prefilled,
                                       self.max_prefill_tokens)
                            r.prefilled += take
                            pieces = [(take, r.dataset)]
                            completed = ([self._waiting.pop(0)]
                                         if r.prefilled >= r.isl else [])
                if pieces:
                    kind = "prefill"
                    n_tok = sum(t for t, _ in pieces)
                    step_s = self.STEP_SCALE * self._get_ttft_chunk(
                        [(t, None, d) for t, d in pieces])
                else:
                    kind = "decode"
                    if self._decode_debt:
                        self._decode_debt -= 1
                    bs = self._decoding_count
                    total_kv = (self._total_active_isl
                                + self._total_generated_tokens)
                    step_s = self.STEP_SCALE * self._get_tpot(
                        bs, total_kv=total_kv,
                        avg_isl=self._total_active_isl / max(bs, 1))

            over = 0.0 if kind == "prefill" else (
                self.DECODE_OVERHEAD_MS
                + self.DECODE_OVERHEAD_MS_PER_REQ * bs) / 1000.0
            t0 = time.monotonic()
            time.sleep(step_s + over)
            t1 = time.monotonic()
            # What the host cost on top of what was asked for. The replay
            # sleeps in real time, so an overloaded or contended host lengthens
            # every step and the trace scores worse for a reason that is not the
            # model -- the same trace once scored 5-7 points apart on a laptop
            # and on the server. Thread count grows with the batch (one blocked
            # thread per in-flight request, ~190 at 26 req/s against ~30 at 8),
            # so if this is load-dependent it belongs in the error budget.
            self._oversleep_s += max(0.0, (t1 - t0) - (step_s + over))
            self._asked_s += step_s + over
            self._overhead_s += over
            self._step_count += 1
            if over > 0 and self.energy_model is not None and kind == "decode":
                # At the step's own power. Measured against the hardware, the
                # deficit at pure decode is 3.6% and the overhead window is
                # 1.5-4.6% of a step, so the device is drawing near its decode
                # rate through it rather than idling -- the CPU work overlaps
                # kernels that are still finishing.
                self._overhead_energy_j += (
                    self.energy_model.decode_step_j(bs, total_kv)
                    * over / step_s if step_s > 0 else 0.0)

            with self._lock:
                if kind == "prefill":
                    self._prefill_events.append(
                        (t0, t1, n_tok, len(completed),
                         float(sum(r.isl * r.isl for r in completed))))
                    self._prefill_busy_s += t1 - t0
                    self._prefill_tokens += n_tok
                    # Window bookkeeping: record the running total, then drop
                    # everything older than one sample before the window edge.
                    self._duty_samples.append((t1, self._prefill_busy_s))
                    edge = t1 - self.DUTY_WINDOW_S
                    while (len(self._duty_samples) > 1
                           and self._duty_samples[1][0] <= edge):
                        self._duty_samples.popleft()
                    finished_at_prefill = []
                    for r in completed:
                        r.ttft_s = t1 - r.t_start
                        r.step_s = t1 - t0
                        r.prefill_bs = len(completed)
                        self._prefill_count -= 1
                        self._total_prefill_isl -= r.isl
                        # The prefill step emits the FIRST token, as vLLM's
                        # does -- that is what TTFT measures. The engine used
                        # to leave `generated` at 0 here and then run `osl`
                        # decode steps, one more than the request needs: an
                        # extra step per request, worth 1.4% of all steps on
                        # the mlperf trace (mean OSL 71), 0.4% on mixed and
                        # 0.16% on long, and 100% on the OSL=1 requests mixed
                        # carries.
                        r.generated = 1
                        self._total_active_isl += r.isl
                        self._total_generated_tokens += 1
                        self._cum_generated += 1
                        if r.generated >= r.osl:
                            # A one-token output is finished by the prefill
                            # step and never decodes at all.
                            r.t_end = t1
                            self._active_count -= 1
                            self._total_active_isl -= r.isl
                            self._total_generated_tokens -= r.generated
                            finished_at_prefill.append(r)
                        else:
                            self._decoding_count += 1
                            self._total_remaining_osl += r.osl - 1
                            self._running.append(r)
                    for r in finished_at_prefill:
                        r.done.set()
                else:
                    done = []
                    for r in self._running:
                        r.generated += 1
                        if r.generated >= r.osl:
                            done.append(r)
                    n = len(self._running)
                    self._total_generated_tokens += n
                    self._total_remaining_osl -= n
                    self._cum_generated += n
                    for r in done:
                        self._running.remove(r)
                        self._decoding_count -= 1
                        self._active_count -= 1
                        self._total_active_isl -= r.isl
                        self._total_generated_tokens -= r.generated
                        r.t_end = t1
                    self._mark()
                    for r in done:
                        r.done.set()
                    continue
                self._mark()

    def process(self, in_tok: int, out_tok: int, dataset: str = "",
                predicted_osl: int = 0) -> dict:
        """Submit one request and block until the engine has finished it.

        The caller's thread no longer does any timing -- it hands the request
        to the slot's engine loop and waits. That is the point: the engine
        decides what runs next, so exclusivity between prefill and decode and
        the batching of queued prompts both fall out of the loop instead of
        being coordinated between request threads.
        """
        # An engine cannot generate past its context window: vLLM caps a
        # request at max_model_len - prompt_len. Without this the simulator
        # happily decodes beyond the window, and those are the most expensive
        # tokens in the run -- they sit at the largest KV. In
        # loadgen/scenarios/mixed_trace_3ds.csv only 2 of 2107 requests are affected, but
        # they carry 1.84% of every decode token in the trace (one is
        # in=30461, out=16384 with 2307 tokens of room). Never fires on the
        # within-node replay traces, which top out at 12512 tokens.
        room = self.max_model_len - in_tok
        if out_tok > room:
            out_tok = max(room, 1)
            with self._lock:
                self._osl_clamped += 1
            if not GPU._clamp_warned:
                GPU._clamp_warned = True
                log.warning("%s: output length clamped to the context window "
                            "(max_model_len=%d). Requests whose prompt plus "
                            "output exceed it are capped, as the engine would "
                            "cap them; the count is in stats() as "
                            "osl_clamped.",
                            self.gpu_id, self.max_model_len)

        t_arrive = time.monotonic()
        if self._semaphore is not None:
            with self._lock:
                self._pending_count += 1
                self._total_pending_isl += in_tok
            self._semaphore.acquire()

        t_start = time.monotonic()
        req = _Req(in_tok, out_tok, dataset, t_start)
        req.pred_osl = int(predicted_osl or 0)
        # Pending -> active in ONE lock section. It used to be two, and between
        # them the request was in neither counter: a PollStats landing in that
        # window returned an (active + pending) short by one, which is the
        # figure every scheduler treats as the datacentre's whole in-flight
        # load. Microseconds wide, but the two counters are documented as
        # disjoint AND exhaustive, and only the second was true.
        with self._lock:
            if self._semaphore is not None:
                self._pending_count -= 1
                self._total_pending_isl -= in_tok
            self._active_count += 1
            self._total_requests += 1
            self._cum_isl += in_tok
            self._prefill_count += 1
            self._total_prefill_isl += in_tok
            self._waiting.append(req)
            self._mark()
            self._ensure_engine()

        try:
            # Bounded, so a slot whose engine died without reaching the handler
            # above still returns instead of pinning a worker thread for the
            # length of the run.
            if not req.done.wait(timeout=self.REQUEST_TIMEOUT_S):
                req.failed = (f"engine did not finish within "
                              f"{self.REQUEST_TIMEOUT_S:.0f}s")
                log.error("%s: %s", self.gpu_id, req.failed)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()
        if req.failed:
            raise RuntimeError(f"{self.gpu_id}: {req.failed}")

        t_end = req.t_end
        # Queue ends at admission to prefill, not at the seat: a prompt holding
        # a seat behind a full KV cache is still queueing. The sums are
        # unchanged -- queue_ms + ttft_ms and queue_ms + service_ms are what
        # they were -- only the boundary between the parts moves.
        t_admit = req.t_admit or t_start
        service_ms = (t_end - t_admit) * 1000.0
        ttft_ms = (req.ttft_s - (t_admit - t_start)) * 1000.0
        # osl - 1, not osl. TTFT covers the first token -- the prefill step
        # emits it -- so the decode phase produces the remaining osl - 1 and
        # that is what the inter-token latency averages over. The divisor was
        # osl while the engine also ran osl decode steps, which was internally
        # consistent and matched no definition of TPOT; both are corrected
        # together.
        avg_tpot_ms = (service_ms - ttft_ms) / max(out_tok - 1, 1)
        return {
            "gpu_id": self.gpu_id,
            "service_ms": service_ms,
            "ttft_ms": ttft_ms,
            "avg_tpot_ms": avg_tpot_ms,
            "queue_ms": (t_admit - t_arrive) * 1000.0,
            "start_ms": t_admit * 1000.0,
            "finish_ms": t_end * 1000.0,
            "prefill_bs": req.prefill_bs,
            "wait_ms": req.wait_s * 1000.0,
            "step_ms": req.step_s * 1000.0,
        }

    async def process_async(self, in_tok: int, out_tok: int,
                            dataset: str = "", predicted_osl: int = 0) -> dict:
        """`process`, for a caller on an event loop.

        DELIBERATELY A COPY, not a refactor. `process` is the path the latency
        model was fitted through and the one `validate_simulation.py` and
        `testbed/phase2_endtoend/replay_sim.py` still call; sharing a body
        would mean editing it, and an edit there invalidates the calibration
        rather than one run. The duplication is held honest by
        test_inference_engine's check that both paths answer the same, which is
        cheaper than the risk of touching the fitted path.

        The two differences are the two waits, and the timing stamps around
        them are placed exactly where the synchronous path puts them --
        `t_arrive` BEFORE waiting for a seat and `t_start` after. That is the
        whole point of this exercise: the seat wait has to land in `queue_ms`.
        Move either stamp and the time disappears into the same hole
        `dc_queue` currently reports it in.
        """
        room = self.max_model_len - in_tok
        if out_tok > room:
            out_tok = max(room, 1)
            with self._lock:
                self._osl_clamped += 1
            if not GPU._clamp_warned:
                GPU._clamp_warned = True
                log.warning("%s: output length clamped to the context window "
                            "(max_model_len=%d). Requests whose prompt plus "
                            "output exceed it are capped, as the engine would "
                            "cap them; the count is in stats() as "
                            "osl_clamped.",
                            self.gpu_id, self.max_model_len)

        t_arrive = time.monotonic()
        if self._semaphore is not None:
            with self._lock:
                self._pending_count += 1
                self._total_pending_isl += in_tok
            await self._semaphore.acquire_async()

        t_start = time.monotonic()
        req = _Req(in_tok, out_tok, dataset, t_start)
        req.pred_osl = int(predicted_osl or 0)
        with self._lock:
            if self._semaphore is not None:
                self._pending_count -= 1
                self._total_pending_isl -= in_tok
            self._active_count += 1
            self._total_requests += 1
            self._cum_isl += in_tok
            self._prefill_count += 1
            self._total_prefill_isl += in_tok
            self._waiting.append(req)
            self._mark()
            self._ensure_engine()

        try:
            if not await req.done.wait_async(timeout=self.REQUEST_TIMEOUT_S):
                req.failed = (f"engine did not finish within "
                              f"{self.REQUEST_TIMEOUT_S:.0f}s")
                log.error("%s: %s", self.gpu_id, req.failed)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()
        if req.failed:
            raise RuntimeError(f"{self.gpu_id}: {req.failed}")

        t_end = req.t_end
        t_admit = req.t_admit or t_start
        service_ms = (t_end - t_admit) * 1000.0
        ttft_ms = (req.ttft_s - (t_admit - t_start)) * 1000.0
        avg_tpot_ms = (service_ms - ttft_ms) / max(out_tok - 1, 1)
        return {
            "gpu_id": self.gpu_id,
            "service_ms": service_ms,
            "ttft_ms": ttft_ms,
            "avg_tpot_ms": avg_tpot_ms,
            "queue_ms": (t_admit - t_arrive) * 1000.0,
            "start_ms": t_admit * 1000.0,
            "finish_ms": t_end * 1000.0,
            "prefill_bs": req.prefill_bs,
            "wait_ms": req.wait_s * 1000.0,
            "step_ms": req.step_s * 1000.0,
        }

    def gpu_energy_j(self) -> float:
        """Energy attributable to the devices, with the node's share removed.

        The only figure a power meter can be compared against: NVML reads the
        GPUs, and HOST_POWER_W is an allocation of everything else on the node.
        """
        return self.energy_j() - self._host_energy_j

    def energy_j(self) -> float:
        """Total energy consumed by this GPU in joules.

        Returns 0 until compute_energy() is called after the experiment.
        """
        with self._lock:
            return self._energy_j

    def idle_energy_j(self) -> float:
        """Of that total, what was drawn with nothing in flight.

        0 on the legacy three-state path, which prices idle inside its
        interval loop and cannot separate it.
        """
        with self._lock:
            return self._idle_energy_j

    def host_energy_j(self) -> float:
        """Of that total, this engine's share of the node's host draw."""
        with self._lock:
            return self._host_energy_j

    def compute_energy(self, sim_start: float, sim_end: float) -> None:
        """Compute energy over [sim_start, sim_end].

        Uses the measured per-step model when this (model, TP) was calibrated,
        and the legacy three-state constants otherwise.
        """
        if self.energy_model is not None:
            return self._compute_energy_measured(sim_start, sim_end)
        return self._compute_energy_three_state(sim_start, sim_end)

    def _compute_energy_measured(self, sim_start: float, sim_end: float) -> None:
        """Integrate the measured per-step energy model over the time series.

        Three kinds of interval, priced by their own term:

          decode   n_steps = delta(generated) / decoding_count, a COUNT rather
                   than an estimate, because every step emits exactly one token
                   per decoding request. Energy closes in one expression since
                   the cache grows linearly inside the interval.
          prefill  one forward pass per recorded chunk, priced on the tokens in
                   that chunk. Decode is paused throughout, so these intervals
                   contribute no decode steps and no double counting arises --
                   which is why this needs none of the subtract-and-replace
                   machinery the three-state version required.
          idle     measured idle draw times duration, for intervals with nothing
                   decoding and no prefill running.

        Elapsed time enters only for idle and for the reported effective power.
        The joules attributed to compute do not depend on the latency model.
        """
        m = self.energy_model
        with self._lock:
            ts = list(self._time_series)
            pfe = sorted(self._prefill_events)

        energy_j = 0.0
        time_at_bs: Dict[int, float] = defaultdict(float)
        decode_steps = 0.0
        decode_tokens = 0

        prefill_seconds = sum(min(pe, sim_end) - max(ps, sim_start)
                              for ps, pe, _n, _nr, _sq in pfe
                              if min(pe, sim_end) > max(ps, sim_start))

        # -- compute intervals between successive marks --
        prev = None
        for entry in ts:
            t = entry[0]
            if prev is not None:
                pt, _pa, pdec, pkv, pgen = prev
                a, b = max(pt, sim_start), min(t, sim_end)
                if b > a:
                    time_at_bs[pdec] += b - a
                    if pdec > 0:
                        # Tokens decode produced in this interval. The cache
                        # also jumps when a prompt lands at the end of a
                        # prefill, so the step count is read from generated
                        # tokens; the cache end point used for the e_kv term
                        # is the one those steps produced.
                        d_gen = entry[4] - pgen
                        if d_gen > 0:
                            e, n = m.decode_segment_j(pdec, pkv, pkv + d_gen)
                            energy_j += e
                            decode_steps += n
                            decode_tokens += d_gen
            prev = entry
            if t > sim_end:
                break

        # -- tail after the last mark --
        if ts:
            lt, _la, ldec, lkv, _lg = ts[-1]
            if sim_end > max(lt, sim_start):
                time_at_bs[ldec] += sim_end - max(lt, sim_start)

        # -- idle: intervals where the engine holds nothing at all --
        # Defined on the active count rather than as "window minus busy",
        # because prefill overlaps intervals whose decoding count is nonzero
        # (decoders are paused, not gone) and subtracting both would count
        # that time twice. Residual time inside a busy interval that neither
        # a decode step nor a prefill chunk claims is left unpriced, which is
        # the safe direction.
        window_s = max(0.0, sim_end - sim_start)
        idle_s = 0.0
        # The span BEFORE the engine's first mark. A mark is written when work
        # arrives, so before the first one the device was powered and empty for
        # all of it. The loop below cannot see that span -- it prices intervals
        # between successive marks and the first mark has no predecessor -- so
        # it went to neither idle nor compute and only the host term covered
        # it. The three-state path opposite prices it (prev_bs = 0 from
        # sim_start), and the two disagreed by exactly idle_w x head seconds:
        # a datacentre whose first request landed 35 s into the window
        # under-reported by 10.3 kJ of 102.8 kJ, 10%.
        #
        # Inert on the gRPC path, where compute_energy_from_duration sets
        # sim_start to this engine's first mark and the span is zero by
        # construction. It bites the callers that pass a window of their own.
        if ts and ts[0][0] > sim_start:
            idle_s += min(ts[0][0], sim_end) - sim_start
        prev_e = None
        for entry in ts:
            if prev_e is not None and prev_e[1] == 0:
                a, b = max(prev_e[0], sim_start), min(entry[0], sim_end)
                if b > a:
                    idle_s += b - a
            prev_e = entry
            if entry[0] > sim_end:
                break
        if ts and ts[-1][1] == 0:
            idle_s += max(0.0, sim_end - max(ts[-1][0], sim_start))
        if not ts:
            idle_s = window_s

        # The tail when the engine was still HOLDING work at the last mark.
        #
        # The branch above credits idle for the span after the last transition,
        # but only when that transition left the engine empty. The series ends
        # at the last recorded transition, so if the window outlives it with
        # work still in flight -- the run was cut before this engine drained --
        # that span was priced as neither decode nor idle and only the host
        # term covered it. The time profile above already counts it; the energy
        # did not.
        #
        # Not small. A datacentre still busy when a 120 s window closed
        # reported 75 kJ against an idle floor of 217 kJ, and across a fleet it
        # made the BUSIEST sites look cheapest -- the exact direction that
        # flatters a policy for consolidating load.
        #
        # Charged at idle, which is a lower bound rather than an estimate:
        # whatever the engine was really doing there, a powered device drew at
        # least this. The warning is the part that matters, because a run that
        # trips it ended before its datacentres drained and its energy is a
        # floor, not a measurement.
        if ts and ts[-1][1] > 0:
            tail_s = max(0.0, sim_end - max(ts[-1][0], sim_start))
            if tail_s > 1e-6:
                idle_s += tail_s
                if not GPU._tail_warned:
                    GPU._tail_warned = True
                    log.warning(
                        "%s: the energy window outlives the recorded series by "
                        "%.1f s of %.1f s, with %d requests still in flight at "
                        "the last mark. That span is charged idle, so this "
                        "run's energy is a LOWER BOUND. Finalize after the "
                        "datacentres drain.",
                        self.gpu_id, tail_s, window_s, ts[-1][1])
        idle_energy_j = m.idle_j(idle_s)
        energy_j += idle_energy_j

        # -- prefill: one step per recorded chunk --
        for ps, pe, n_tok, n_reqs, sum_sq in pfe:
            if min(pe, sim_end) <= max(ps, sim_start):
                continue
            energy_j += m.prefill_step_j(n_tok, n_reqs, sum_sq_tokens=sum_sq)

        # -- host draw, unchanged: constant across the window --
        self._host_energy_j = self._host_power_share_w * window_s
        self._idle_energy_j = idle_energy_j
        energy_j += self._overhead_energy_j
        energy_j += self._host_energy_j

        with self._lock:
            self._energy_j = energy_j
            self._time_at_bs = time_at_bs
            self._prefill_seconds = prefill_seconds
            self._decode_steps = decode_steps
            self._decode_tokens = decode_tokens
            self._idle_seconds = idle_s

    def _compute_energy_three_state(self, sim_start: float, sim_end: float) -> None:
        """Legacy three-state integral, for models with no measured artifact.

        States:
            idle    → IDLE_POWER_W            (60 W)
            prefill → PREFILL_POWER_W         (400 W) — any time a prefill
                                                       chunk is executing,
                                                       regardless of concurrent
                                                       decoders (which are
                                                       paused during prefill)
            decode  → power_w(bs)             (log-linear in BS from the
                                                       analytical/lookup model)

        Implementation is two-pass:

          Pass 1: walk the BS time series and accumulate (idle | decode)
                  energy, treating the whole window as if no prefill ever
                  ran. This over-attributes decode/idle power to segments
                  that were actually prefill.

          Pass 2: for each recorded prefill event, subtract the Pass-1
                  energy that was wrongly credited to that segment and
                  add the correct PREFILL_POWER_W × duration instead.

        This yields a correct three-state integral without needing an
        expensive merged event list.
        """
        with self._lock:
            ts = list(self._time_series)
            pfe = sorted(self._prefill_events)

        # ── Pass 1: naive bs-based energy (idle or decode) ──
        energy_j = 0.0
        time_at_bs: Dict[int, float] = defaultdict(float)
        prev_t = sim_start
        prev_bs = 0

        for t, bs, _dec, _kv, _gen in ts:
            t = max(t, sim_start)
            if t > sim_end:
                break
            dt = t - prev_t
            if dt > 0:
                power = self._get_power_w(prev_bs) if prev_bs > 0 else self._idle_power_w
                energy_j += power * dt
                time_at_bs[prev_bs] += dt
            prev_t = t
            prev_bs = bs

        dt = sim_end - prev_t
        if dt > 0:
            power = self._get_power_w(prev_bs) if prev_bs > 0 else self._idle_power_w
            energy_j += power * dt
            time_at_bs[prev_bs] += dt

        # ── Pass 2: prefill overlay ──
        # For each prefill chunk, find what bs was "seen" by Pass 1 during
        # that segment, undo that energy contribution, and replace with
        # PREFILL_POWER_W.
        prefill_seconds = 0.0
        for pstart, pend, _n, _nr, _sq in pfe:
            pstart_c = max(pstart, sim_start)
            pend_c = min(pend, sim_end)
            if pend_c <= pstart_c:
                continue
            # Split the prefill segment at every bs transition so we undo
            # the right power on every sub-interval.
            sub_start = pstart_c
            bs_at = 0
            # Find bs in effect at sub_start (last transition <= sub_start)
            for t, bs, _dec, _kv, _gen in ts:
                if t > sub_start:
                    break
                bs_at = bs
            for t, bs, _dec, _kv, _gen in ts:
                if t <= sub_start:
                    continue
                if t >= pend_c:
                    break
                sub_dt = t - sub_start
                if sub_dt > 0:
                    wrong_power = (self._get_power_w(bs_at)
                                   if bs_at > 0 else self._idle_power_w)
                    energy_j += (self._prefill_power_w - wrong_power) * sub_dt
                    time_at_bs[bs_at] -= sub_dt
                sub_start = t
                bs_at = bs
            # Final sub-segment from sub_start to pend_c
            sub_dt = pend_c - sub_start
            if sub_dt > 0:
                wrong_power = (self._get_power_w(bs_at)
                               if bs_at > 0 else self._idle_power_w)
                energy_j += (self._prefill_power_w - wrong_power) * sub_dt
                time_at_bs[bs_at] -= sub_dt
            prefill_seconds += (pend_c - pstart_c)

        # ── Host overlay: constant draw for the full sim window. ──
        # host_share_per_pair × duration is invariant of GPU activity.
        sim_window_s = max(0.0, sim_end - sim_start)
        self._host_energy_j = self._host_power_share_w * sim_window_s
        energy_j += self._overhead_energy_j
        energy_j += self._host_energy_j

        with self._lock:
            self._energy_j = energy_j
            self._time_at_bs = time_at_bs
            self._prefill_seconds = prefill_seconds

    def time_at_bs_snapshot(self) -> Dict[int, float]:
        """Return a copy of cumulative seconds spent at each batch size."""
        with self._lock:
            return dict(self._time_at_bs)

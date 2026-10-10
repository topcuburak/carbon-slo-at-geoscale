# The drain campaign that produced the lookup tables

Everything in `../table_data/` came from here. The scripts are kept so that
returning to the lookup backend does not mean starting from nothing — the
methodology, the datasets, the batch-size ladder and the warm-up handling are
all decided in these files.

**This is the April 2026 campaign, not the current one.** The measurements the
paper rests on are the analytical fits in `../../analytical/fit_data/`,
produced by `testbed/phase1_calibration/` in August and September 2026 on our testbed.
The two are separate programmes and should not be mixed: this one drives vLLM's
`LLMEngine` directly and records per-step CSVs to interpolate; that one fits
closed-form coefficients per (model, TP). The sweep runs on the analytical side.

## What is here

    build_lookups.py          drain CSVs -> ../table_data/*.json. The fitting
                              step; the counterpart of phase1_calibration's
                              fit_latency.py on the analytical side.
    run_*_drain.py            the campaigns, one per dataset and model.
                              humaneval / mbpp / govreport x
                              {llama3_8B, mixtral, deepseek-moe, qwen, 70b}
    run_ttft_sweep.py         prefill sweep
    run_tpot_drain.py         decode drain
    run_pilot.py              a 4 x BS=8 smoke run, worth doing first
    run_batch.py              the shared batching driver
    consolidate.py            merges per-run CSVs
    power_meas.py             nvidia-smi sampling alongside a run
    setup_env.sh              the vLLM environment these were measured under
    reference_outputs/        the small summaries and fits, kept as a record of
                              what a good run looked like
    run_logs/                 stdout from the April runs -- batch sizes, engine
                              settings, vLLM warnings. Not an input to anything;
                              where to look if a table seems wrong

## Where the raw data is

The per-step CSVs are hundreds of megabytes and are NOT in the repository
(`.gitignore`). They live on our testbed, and a local copy is kept archived:

    datacenter/perf_models/lookup/measured/archives/mixtral_deepseek_drains.tar.gz          44M, 468M unpacked
        mixtral_8x7b at TP=4 and deepseek-moe-16b at TP=2 -- the source of four
        of the six tables in ../table_data/.

    datacenter/perf_models/lookup/measured/archives/llama3_8B_drains/*.tar.gz                     27M
        the llama3_8B drains, the decode drain, the prefill sweep and the pilot.

To rebuild the tables:

    tar xzf datacenter/perf_models/lookup/measured/archives/mixtral_deepseek_drains.tar.gz -C /tmp/drains
    DRAIN_ROOT=/tmp/drains/results \
        python datacenter/perf_models/lookup/campaign/build_lookups.py

`DRAIN_ROOT` is why the archives can be unpacked anywhere. See
the archives/ README for what each one holds.

## Status

The lookup backend is reachable — `perf_models/factory.py`, and
`lookup/validation/` and `scripts/benchmarks/single_dc.py` use it — but the sweep does not.
Whether it stays is SIMULATOR_TODO's open question. These scripts are the reason
that question can be answered by re-measuring rather than by guessing.

Note the four `run_*_drain.py` files that also exist in
`testbed/phase1_calibration/` have diverged there by 16 to 44 lines; the testbed
copies are the current ones for llama, and these are the originals the mixtral,
deepseek, qwen and 70b runs were made with.

---

# The original campaign README, kept verbatim

Written when these scripts were the current measurement programme. The paths in
it have been updated to where the scripts now live; nothing else is changed,
because the methodology is the part worth keeping.

Re-runs LLaMA-3.1-8B-Instruct benchmarks on the 4×A100 box with
per-engine-step timing instrumentation, so we can derive per-step TPOT
as a function of `(active_BS, total_KV)` without the noise that
contaminated the v1 benchmarks.

## Why a rewrite

The existing `datacenter/perf_models/lookup/measured/full/llama3.1-8B/*/benchmarks/merged_benchmark.csv`
data only has per-request `(ttft, tpot, inf_latency)` averages. Reconstructing
per-segment TPOT from those aggregates fails because vLLM's continuous
batching with chunked prefill produces per-request TPOT values that are
polluted by concurrent prefill chunks from other requests.

This benchmark fixes the issue by:

1. Running with `enable_chunked_prefill=False` — prefill finishes for all
   requests in one forward pass before decode starts.
2. Using `LLMEngine` directly and timing every `engine.step()` call.
3. Logging per-iteration `(step_type, n_prefilling, n_decoding, step_ms,
   total_kv, active_ids)` so we can track the actual drain timeline.
4. Running TP=4 to match our production target (multi-GPU deployment).

## Prerequisites

- 4×A100-40GB box with NVIDIA driver ≥ 535 (supports CUDA 12.1 runtime)
- Python 3.11
- Miniconda / Anaconda
- `HF_TOKEN` for downloading LLaMA-8B weights

## Setup

```bash
# On the 4×A100 box (our testbed)
cd <repo>

# One-time env setup
export HF_TOKEN=hf_xxx...
bash datacenter/perf_models/lookup/campaign/setup_env.sh
```

The setup script creates a `llm_bench` conda env with `vllm==0.6.3.post1`
and `torch==2.4.0+cu121`, downloads the LLaMA-8B weights, and runs a
smoke test.

## Pilot run

Runs 4 back-to-back batches of BS=8 GovReport to verify the per-step
logging is working.

```bash
conda activate llm_bench
cd <repo>
python datacenter/perf_models/lookup/campaign/run_pilot.py
```

Expected runtime: ~3–5 minutes (engine load + 4 batches).

Outputs land in:

```
datacenter/perf_models/lookup/measured/archives/ (formerly benchmarks_v2/results/)pilot_govreport_bs8/
├── batch_0/
│   ├── per_step.csv       # one row per engine.step() call
│   ├── per_request.csv    # one row per request (ttft, decode_total, osl_generated, ...)
│   └── config.json
├── batch_1/
├── batch_2/
├── batch_3/
└── summary.txt
```

Inspect with:

```bash
head datacenter/perf_models/lookup/measured/archives/ (formerly benchmarks_v2/results/)pilot_govreport_bs8/batch_0/per_step.csv
cat datacenter/perf_models/lookup/measured/archives/ (formerly benchmarks_v2/results/)pilot_govreport_bs8/summary.txt
```

## What to look for in the per_step.csv

The key columns:

| column | meaning |
|---|---|
| `iter` | engine step index (0, 1, 2, ...) |
| `step_type` | `prefill`, `decode`, `mixed`, or `idle` |
| `n_prefilling` | requests still in prefill phase at start of step |
| `n_decoding` | requests in decode phase |
| `step_ms` | measured wall time of this step |
| `total_kv_before` | sum of (ctx_len) over all active seqs, before step |
| `prefill_tokens` | # prefill tokens processed this step |
| `decode_tokens` | # decode tokens generated this step |

For a healthy BS=8 batch with non-chunked prefill:

- **Iteration 0**: `step_type=prefill`, `n_prefilling=8`, `n_decoding=0`,
  `step_ms` = maybe 500–2000 ms (one large prefill pass).
- **Iterations 1..N**: `step_type=decode`, `n_decoding=8 → 7 → ... → 1`
  (draining), `step_ms` = 10–30 ms each.
- **No `mixed` rows** should appear if chunked prefill is genuinely off.

If we see `mixed` rows, that means vLLM admitted a prefill chunk mid-decode
despite our config — we'd need to investigate the EngineArgs.

## Per-request summary

`per_request.csv` has one row per request:

| column | meaning |
|---|---|
| `req_id` | e.g. `batch_0_req3` |
| `isl` | input token count |
| `osl_requested` | max_new_tokens asked |
| `osl_generated` | actual output length |
| `ttft_ms` | first token latency (arrival → first token) |
| `decode_total_ms` | first token → final token |
| `decode_steps` | # engine steps where this req generated a token |
| `finish_reason` | `length` (hit max) or `stop` (EOS) |

**Sanity checks**:

- `ttft_ms` should be nearly identical across all 8 requests in one batch
  (they all finish prefill in the same step).
- `decode_total_ms` should scale with `osl_generated` — short outputs
  finish first (part of the drain).
- `decode_steps` should equal `osl_generated` exactly (one step per token
  in pure decode).

If `ttft_ms` varies wildly across requests, chunked prefill is still happening.
If `decode_steps > osl_generated`, vLLM is double-counting — needs debug.

## Next steps after the pilot

Once the pilot output looks clean:

1. **Scale up BS** — run BS ∈ {1, 2, 4, 8, 16, 32, 64, 128} for GovReport,
   30–50 batch instances each.
2. **Add more datasets** — Arxiv, ShareGPT, MBPP, HumanEval.
3. **Build the lookup table** — ingest all per_step.csv files, bin by
   `(BS, total_KV)`, compute the per-step TPOT curve.
4. **Plug into `inference_engine.py`** — replace the analytical TPOT model with
   the new lookup.

## Troubleshooting

**OOM during prefill**: the total ISL of the batch exceeds what fits in
one forward pass at the current `max_num_batched_tokens` + `max_num_seqs`.
Lower the batch size or reduce `gpu_memory_utilization` in `run_batch.py`.

**`mixed` step_type appears**: vLLM decided to admit a prefill chunk
mid-decode. Check that `enable_chunked_prefill=False` actually took effect —
earlier vllm versions had a bug where the flag was silently ignored.

**Model download fails**: ensure `HF_TOKEN` is exported and you've accepted
the LLaMA-3.1 license on HuggingFace.

**`vllm` import errors after install**: driver/CUDA mismatch. `setup_env.sh`
pins CUDA 12.1 wheels which work with drivers 525+. If your driver is older,
downgrade PyTorch/vLLM to CUDA 11.8 wheels.

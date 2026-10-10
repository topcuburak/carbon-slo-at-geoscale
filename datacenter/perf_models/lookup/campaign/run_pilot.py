#!/usr/bin/env python3
"""Pilot: run 4 back-to-back BS=8 GovReport batches and print sanity stats.

Loads the GovReport execution log (which has prompt text), slices out
4 non-overlapping batches of 8 requests each, reuses the same LLMEngine
for all 4, and prints per-batch + aggregate summaries.

Usage:
    conda activate llm_bench
    python datacenter/perf_models/lookup/campaign/run_pilot.py

Outputs written to:
    datacenter/perf_models/lookup/measured/archives/unpacked/pilot_govreport_bs8/batch_{0..3}/
        per_step.csv
        per_request.csv
        config.json
    datacenter/perf_models/lookup/measured/archives/unpacked/pilot_govreport_bs8/summary.txt
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make this directory importable (run_batch lives beside us)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402

from run_batch import run_batch, _load_prompts_from_execlog  # noqa: E402


# ────────────────────────── config ──────────────────────────

EXECLOG = Path(
    "datacenter/perf_models/lookup/measured/full/llama3.1-8B/GovReport/execution_logs/"
    "meta-llama_Meta-Llama-3.1-8B-Instruct__govreport__BS_32.csv"
)
N_BATCHES = 4
BS = 8
# 0 = no cap — let vLLM generate until natural EOS (bounded only by max_model_len)
MAX_OUTPUT_TOKENS = 0
# Tensor parallel size. Override with PILOT_TP=2 env var if GPU 0 is busy.
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/pilot_govreport_bs8")


def main():
    if not EXECLOG.exists():
        print(f"ERROR: execlog not found: {EXECLOG}")
        print("Make sure you're running from the repo root.")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Pre-load 4 non-overlapping slices of 8 prompts ──
    batches = []
    df = pd.read_csv(EXECLOG)
    if len(df) < BS * N_BATCHES:
        print(f"ERROR: execlog has only {len(df)} rows, need {BS * N_BATCHES}")
        sys.exit(1)

    for b in range(N_BATCHES):
        start = b * BS
        sub = df.iloc[start:start + BS]
        prompts = sub["prompt"].astype(str).tolist()
        isls = sub["in_tokens"].astype(int).tolist()
        batches.append((f"batch_{b}", prompts, isls))

    # ── Build the engine once, share across all 4 batches ──
    # Fixed max_model_len=64K accommodates the longest GovReport prompts
    # (~32K) plus natural-EOS outputs up to ~32K. vLLM requires
    # max_num_batched_tokens >= max_model_len; we also need it to cover
    # the biggest batch's total prefill in one pass.
    total_isl_max = max(sum(isls) for _, _, isls in batches)
    max_model_len = 65536
    max_num_batched = max(total_isl_max + 1, max_model_len)

    print("=" * 72)
    print("  drain campaign pilot: 4 × BS=8 GovReport")
    print("=" * 72)
    print(f"  Execlog:            {EXECLOG}")
    print(f"  N batches:          {N_BATCHES}")
    print(f"  BS per batch:       {BS}")
    print(f"  Max output tokens:  {MAX_OUTPUT_TOKENS}")
    print(f"  Model:              {MODEL}")
    print(f"  TP:                 {TP}")
    print(f"  max_num_batched:    {max_num_batched}")
    print(f"  max_model_len:      {max_model_len}")
    print("=" * 72)

    # Preview batch shapes
    for bid, prompts, isls in batches:
        print(f"  {bid}: n={len(prompts)} total_ISL={sum(isls):>6}  "
              f"ISL range [{min(isls):>5}, {max(isls):>5}]")
    print()

    print(f"[pilot] Creating LLMEngine (TP={TP})...")
    t_engine_start = time.monotonic()
    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="float16",
        max_num_seqs=BS,
        max_num_batched_tokens=max_num_batched,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=42,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    print(f"[pilot] Engine ready in {time.monotonic() - t_engine_start:.1f}s\n")

    # ── Run each batch ──
    results = []
    for batch_id, prompts, isls in batches:
        t0 = time.monotonic()
        print(f"[pilot] Running {batch_id}...")
        trackers, per_step = run_batch(
            prompts=prompts,
            isls=isls,
            batch_id=batch_id,
            out_dir=OUT_ROOT / batch_id,
            model=MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tensor_parallel_size=TP,
            engine=engine,
            verbose=False,
        )
        wall_s = time.monotonic() - t0

        # Summarize the batch
        n_iter = len(per_step)
        n_prefill = sum(1 for s in per_step if s["step_type"] == "prefill")
        n_decode = sum(1 for s in per_step if s["step_type"] == "decode")
        n_mixed = sum(1 for s in per_step if s["step_type"] == "mixed")

        ttfts = [t.first_token_time_ms - t.arrival_time_ms
                 for t in trackers.values()
                 if t.first_token_time_ms is not None]
        gen_lens = [t.osl_generated for t in trackers.values()]

        print(f"  {batch_id} done in {wall_s:.1f}s  "
              f"iters={n_iter} (prefill={n_prefill} decode={n_decode} mixed={n_mixed})")
        if ttfts:
            print(f"    ttft_ms: min={min(ttfts):.0f} "
                  f"max={max(ttfts):.0f} mean={sum(ttfts)/len(ttfts):.0f}")
        print(f"    osl_generated: min={min(gen_lens)} "
              f"max={max(gen_lens)} mean={sum(gen_lens)/len(gen_lens):.0f}")

        results.append({
            "batch_id": batch_id,
            "wall_s": round(wall_s, 2),
            "iters": n_iter,
            "prefill_steps": n_prefill,
            "decode_steps": n_decode,
            "mixed_steps": n_mixed,
            "n_requests": len(trackers),
            "mean_gen_len": round(sum(gen_lens)/len(gen_lens), 1),
        })

    # ── Aggregate summary ──
    summary_path = OUT_ROOT / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("drain campaign pilot: 4 x BS=8 GovReport\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"{'batch':<12} {'wall_s':>8} {'iters':>7} "
                f"{'prefill':>8} {'decode':>7} {'mixed':>6} {'mean_gen':>9}\n")
        f.write("-" * 72 + "\n")
        for r in results:
            f.write(f"{r['batch_id']:<12} {r['wall_s']:>8.1f} "
                    f"{r['iters']:>7} {r['prefill_steps']:>8} "
                    f"{r['decode_steps']:>7} {r['mixed_steps']:>6} "
                    f"{r['mean_gen_len']:>9.1f}\n")

    print()
    print(f"[pilot] Summary: {summary_path}")
    print(f"[pilot] Per-batch artifacts: {OUT_ROOT}/batch_*/")
    print()
    print("[pilot] Inspect per-step logs for one batch with:")
    print(f"    head {OUT_ROOT}/batch_0/per_step.csv")
    print(f"    head {OUT_ROOT}/batch_0/per_request.csv")


if __name__ == "__main__":
    main()

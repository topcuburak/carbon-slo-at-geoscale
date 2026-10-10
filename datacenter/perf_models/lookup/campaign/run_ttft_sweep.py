#!/usr/bin/env python3
"""TTFT sweep: 128 GovReport prompts × {BS=1, 2, 4, 8}.

For each batch size, splits the same 128 prompts into ceil(128/BS)
groups and runs each group as a separate inference (single engine
instance, reused across groups).

Measures per-group TTFT (identical across requests in the group under
single-pass prefill) to check how TTFT scales with total_ISL and BS.

Output: datacenter/perf_models/lookup/measured/archives/unpacked/ttft_sweep/
    group_bs{BS}_{idx}/
        per_step.csv, per_request.csv, config.json
    sweep_summary.csv
        bs, group_idx, n_requests, total_isl, ttft_ms, prefill_step_ms,
        per_request_isls

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=2,3 PILOT_TP=2 python datacenter/perf_models/lookup/campaign/run_ttft_sweep.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from run_batch import run_batch  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402


# ── config ──────────────────────────────────────────────────────────────

EXECLOG = Path(
    "datacenter/perf_models/lookup/measured/full/llama3.1-8B/GovReport/execution_logs/"
    "meta-llama_Meta-Llama-3.1-8B-Instruct__govreport__BS_32.csv"
)
N_PROMPTS = 128                      # fixed subset for all BS values
BS_VALUES = [1, 2, 4, 8, 16]  # TP=2 ceiling: BS>=32 can't fit single-pass prefill
MAX_OUTPUT_TOKENS = 32               # cap output so we only measure prefill
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/ttft_sweep")


def main():
    if not EXECLOG.exists():
        print(f"ERROR: execlog not found: {EXECLOG}")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXECLOG)
    if len(df) < N_PROMPTS:
        print(f"ERROR: execlog has {len(df)} rows, need {N_PROMPTS}")
        sys.exit(1)
    df = df.iloc[:N_PROMPTS].reset_index(drop=True)
    all_prompts = df["prompt"].astype(str).tolist()
    all_isls = df["in_tokens"].astype(int).tolist()

    print("=" * 72)
    print("  TTFT sweep: 128 GovReport prompts")
    print("=" * 72)
    print(f"  Prompts:      {N_PROMPTS}")
    print(f"  BS values:    {BS_VALUES}")
    print(f"  Max output:   {MAX_OUTPUT_TOKENS} tokens (TTFT-focused)")
    print(f"  TP:           {TP}")
    print(f"  Longest ISL:  {max(all_isls)}")
    print(f"  Total ISL:    {sum(all_isls):,}")
    print("=" * 72)

    # ── build one engine sized for the largest prefill we'll actually need ──
    # On TP=2 (2×40GB A100), the KV cache reservation is sized by
    #     max_num_seqs × max_model_len × ~4KB/token/layer × num_layers / TP
    # For LLaMA-8B with 32 layers, max_model_len=16384 and max_num_seqs=32
    # reserves ~32 GB per GPU, which fits.
    #
    # Activations for one forward pass are driven by max_num_batched_tokens.
    # We cap it at the actual worst-case group total_ISL we'll run.
    max_model_len = 16384  # per-sequence ceiling (longest GovReport prompt < 12K)
    longest_isl = max(all_isls)
    if longest_isl >= max_model_len:
        print(f"ERROR: longest ISL {longest_isl} exceeds max_model_len {max_model_len}")
        sys.exit(1)

    worst_group_total_isl = 0
    for bs in BS_VALUES:
        n_groups = N_PROMPTS // bs
        for g in range(n_groups):
            start = g * bs
            group_total = sum(all_isls[start:start + bs])
            if group_total > worst_group_total_isl:
                worst_group_total_isl = group_total
    max_num_batched = max(worst_group_total_isl + 1, max_model_len)
    print(f"[sweep] Worst-case group total_ISL across all BS values: "
          f"{worst_group_total_isl:,}")

    print(f"\n[sweep] Building engine (TP={TP}, "
          f"max_num_batched={max_num_batched}, max_model_len={max_model_len})")
    t0 = time.monotonic()
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="float16",
        max_num_seqs=max(BS_VALUES),
        max_num_batched_tokens=max_num_batched,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=42,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    ))
    print(f"[sweep] Engine ready in {time.monotonic() - t0:.1f}s\n")

    # ── sweep ──
    summary_rows = []
    total_groups = sum((N_PROMPTS + bs - 1) // bs for bs in BS_VALUES)
    group_counter = 0

    for bs in BS_VALUES:
        n_groups = N_PROMPTS // bs
        print(f"── BS={bs}: {n_groups} groups ──")
        bs_t0 = time.monotonic()

        for g in range(n_groups):
            group_counter += 1
            start = g * bs
            prompts = all_prompts[start:start + bs]
            isls = all_isls[start:start + bs]
            total_isl = sum(isls)
            batch_id = f"group_bs{bs}_{g:03d}"

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

            prefill_step_ms = per_step[0]["step_ms"] if per_step else 0.0
            any_tracker = next(iter(trackers.values()))
            ttft_ms = (any_tracker.first_token_time_ms
                       - any_tracker.arrival_time_ms) if any_tracker.first_token_time_ms else 0.0

            summary_rows.append({
                "bs": bs,
                "group_idx": g,
                "n_requests": len(prompts),
                "total_isl": total_isl,
                "ttft_ms": round(ttft_ms, 3),
                "prefill_step_ms": round(prefill_step_ms, 3),
                "per_request_isls": "|".join(str(i) for i in isls),
            })

        bs_elapsed = time.monotonic() - bs_t0
        avg_ttft = sum(r["ttft_ms"] for r in summary_rows if r["bs"] == bs) / n_groups
        print(f"  done in {bs_elapsed:.1f}s  avg TTFT={avg_ttft:.1f} ms")

    # ── write summary CSV ──
    summary_csv = OUT_ROOT / "sweep_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bs", "group_idx", "n_requests", "total_isl",
            "ttft_ms", "prefill_step_ms", "per_request_isls",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[sweep] Wrote {summary_csv}")
    print(f"[sweep] {group_counter} groups × {len(BS_VALUES)} BS values done")

    # Per-BS aggregate
    print("\nPer-BS aggregate TTFT:")
    print(f"  {'BS':>4} {'n_groups':>9} {'avg_total_isl':>14} {'avg_ttft_ms':>13}")
    for bs in BS_VALUES:
        rows = [r for r in summary_rows if r["bs"] == bs]
        if not rows:
            continue
        avg_isl = sum(r["total_isl"] for r in rows) / len(rows)
        avg_ttft = sum(r["ttft_ms"] for r in rows) / len(rows)
        print(f"  {bs:>4} {len(rows):>9} {avg_isl:>14,.0f} {avg_ttft:>12.1f}")


if __name__ == "__main__":
    main()

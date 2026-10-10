#!/usr/bin/env python3
"""TPOT drain sweep: same 128 GovReport prompts, BS={1,2,4,8,16}, natural EOS.

For each batch size, splits the same 128 prompts into ceil(128/BS) groups
(identical to the TTFT sweep) and runs each as a separate inference with
max_tokens=0 (unrestricted) so requests finish at their natural EOS. This
produces drain dynamics: as each request hits EOS, alive count drops and
total_KV shrinks, letting us observe per-step TPOT at many (BS, total_KV)
states from a single group.

Per-step log (via run_batch.py) records:
  iter, step_type, n_prefilling, n_decoding, n_active, step_ms,
  total_kv_before, total_kv_after, prefill_tokens, decode_tokens,
  active_ids

Per-request log records:
  req_id, isl, osl_requested, osl_generated, ttft_ms, decode_total_ms,
  decode_steps, arrival_iter, first_token_iter, final_iter, finish_reason

Output: datacenter/perf_models/lookup/measured/archives/unpacked/tpot_drain/
    group_bs{BS}_{idx}/
        per_step.csv, per_request.csv, config.json
    drain_summary.csv
        bs, group_idx, n_requests, total_isl, n_iters,
        prefill_ms, min_osl, max_osl, mean_osl, n_distinct_bs_states

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=2,3 PILOT_TP=2 python datacenter/perf_models/lookup/campaign/run_tpot_drain.py
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
from consolidate import consolidate  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402


# ── config (mirrors run_ttft_sweep.py for 1:1 comparability) ──────────

EXECLOG = Path(
    "datacenter/perf_models/lookup/measured/full/llama3.1-8B/GovReport/execution_logs/"
    "meta-llama_Meta-Llama-3.1-8B-Instruct__govreport__BS_32.csv"
)
N_PROMPTS = 128                        # same subset as TTFT sweep
BS_VALUES = [1, 2, 4, 8, 16]           # same as TTFT sweep
# Cap at 4K tokens. GovReport natural outputs are typically 300-2000
# tokens, so this leaves headroom without letting runaway generation
# block the sweep. Stop strings in run_batch.py catch common repetition
# patterns as a second safety net.
MAX_OUTPUT_TOKENS = 4096
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/tpot_drain")


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
    print("  TPOT drain sweep: 128 GovReport prompts, natural EOS")
    print("=" * 72)
    print(f"  Prompts:      {N_PROMPTS}")
    print(f"  BS values:    {BS_VALUES}")
    print(f"  max_tokens:   {MAX_OUTPUT_TOKENS} (0 = natural EOS)")
    print(f"  TP:           {TP}")
    print(f"  Longest ISL:  {max(all_isls)}")
    print(f"  Total ISL:    {sum(all_isls):,}")
    print("=" * 72)

    # ── engine sizing: matches TTFT sweep to keep results directly
    #     comparable. max_model_len=16384 caps per-sequence context
    #     (longest GovReport ISL ~12K + natural output ~4K fits). ──
    max_model_len = 16384
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
    print(f"[drain] worst_group_total_isl: {worst_group_total_isl:,}")

    print(f"\n[drain] Building engine (TP={TP}, "
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
    print(f"[drain] Engine ready in {time.monotonic() - t0:.1f}s\n")

    # ── sweep ──
    summary_rows = []
    total_groups = sum(N_PROMPTS // bs for bs in BS_VALUES)
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

            # Stats about the drain
            osls = [t.osl_generated for t in trackers.values()]
            prefill_ms = per_step[0]["step_ms"] if per_step else 0.0
            # Unique alive counts seen during decode (excludes prefill iter 0)
            decode_states = {s["n_decoding"] for s in per_step[1:]
                             if s["n_decoding"] > 0}
            n_distinct_bs_states = len(decode_states)

            summary_rows.append({
                "bs": bs,
                "group_idx": g,
                "n_requests": len(prompts),
                "total_isl": total_isl,
                "n_iters": len(per_step),
                "prefill_ms": round(prefill_ms, 3),
                "min_osl": min(osls) if osls else 0,
                "max_osl": max(osls) if osls else 0,
                "mean_osl": round(sum(osls) / len(osls), 1) if osls else 0.0,
                "n_distinct_bs_states": n_distinct_bs_states,
            })

            if (g + 1) % max(1, n_groups // 4) == 0 or g == n_groups - 1:
                print(f"  [{group_counter}/{total_groups}] "
                      f"bs={bs} group={g+1}/{n_groups}  "
                      f"iters={summary_rows[-1]['n_iters']:>5}  "
                      f"osl=[{summary_rows[-1]['min_osl']:>4},"
                      f"{summary_rows[-1]['max_osl']:>4}]  "
                      f"states={n_distinct_bs_states}")

        bs_elapsed = time.monotonic() - bs_t0
        print(f"  done in {bs_elapsed:.1f}s")

    # ── write summary ──
    summary_csv = OUT_ROOT / "drain_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bs", "group_idx", "n_requests", "total_isl", "n_iters",
            "prefill_ms", "min_osl", "max_osl", "mean_osl",
            "n_distinct_bs_states",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[drain] Wrote {summary_csv}")
    print(f"[drain] {group_counter} groups done")

    # Consolidate per-group CSVs into per-BS CSVs
    print("\n[drain] Consolidating per-group CSVs into per-BS files...")
    consolidate(OUT_ROOT)

    print("\nPer-BS aggregate:")
    print(f"  {'BS':>4} {'groups':>7} {'avg_iters':>11} {'avg_mean_osl':>14} "
          f"{'avg_states':>12}")
    for bs in BS_VALUES:
        rows = [r for r in summary_rows if r["bs"] == bs]
        if not rows:
            continue
        avg_iters = sum(r["n_iters"] for r in rows) / len(rows)
        avg_mean_osl = sum(r["mean_osl"] for r in rows) / len(rows)
        avg_states = sum(r["n_distinct_bs_states"] for r in rows) / len(rows)
        print(f"  {bs:>4} {len(rows):>7} {avg_iters:>11.0f} "
              f"{avg_mean_osl:>14.0f} {avg_states:>12.1f}")


if __name__ == "__main__":
    main()

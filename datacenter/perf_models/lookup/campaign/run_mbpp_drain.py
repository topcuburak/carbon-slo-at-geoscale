#!/usr/bin/env python3
"""MBPP TPOT drain sweep — all 974 problems x BS={1,2,4,8,16,32,64,128}.

MBPP is short-form Python function generation. Each request is a
1-2 sentence task description plus one example assertion, and the
model generates a Python function until natural EOS. ISLs are tiny
(~50-150 tokens) and OSLs are naturally varied (~40-200 tokens),
giving us a large pool of clean drain data to complement HumanEval.

Uses the same BS sweep structure and engine config as
run_humaneval_drain.py so the two datasets are directly comparable.

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=2,3 PILOT_TP=2 \\
        python datacenter/perf_models/lookup/campaign/run_mbpp_drain.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402

from run_batch import run_batch  # noqa: E402
from consolidate import consolidate  # noqa: E402


# ── config ──────────────────────────────────────────────────────────────

DATASET = "google-research-datasets/mbpp"
CONFIG = "full"
# Merge all splits for maximum coverage: 374 + 500 + 90 + 10 = 974.
SPLITS = ["train", "test", "validation", "prompt"]
N_PROMPTS = 974
BS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
# MBPP solutions are typically short (<200 tokens). 512 leaves ample
# headroom so natural EOS dominates without wasted compute on outliers.
MAX_OUTPUT_TOKENS = 512
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/mbpp_drain")


def load_mbpp_prompts(tokenizer, n: int):
    """Load MBPP problems from all splits, wrap each as an instruction prompt.

    MBPP 'full' config fields: task_id, text, code, test_list,
    test_setup_code, challenge_test_list. The 'text' field is a
    short task description. Common practice is to include the first
    assertion from test_list as a hint so the function signature
    is unambiguous.
    """
    rows = []
    for split in SPLITS:
        ds = load_dataset(DATASET, CONFIG, split=split)
        for row in ds:
            rows.append(row)
            if len(rows) >= n:
                break
        if len(rows) >= n:
            break

    prompts = []
    isls = []
    for row in rows:
        task_text = row["text"]
        test_list = row.get("test_list") or []
        first_test = test_list[0] if test_list else ""
        hint = f"\n\nYour function must pass this test:\n{first_test}" if first_test else ""
        messages = [
            {
                "role": "user",
                "content": (
                    "Write a Python function for the following task. "
                    "Only output the function definition."
                    f"\n\n{task_text}{hint}"
                ),
            },
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tok_len = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
        prompts.append(chat_prompt)
        isls.append(tok_len)
        if len(prompts) >= n:
            break
    return prompts, isls


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  MBPP TPOT drain sweep")
    print("=" * 72)
    print(f"  Dataset:     {DATASET} ({CONFIG}, splits={SPLITS})")
    print(f"  N prompts:   {N_PROMPTS}")
    print(f"  BS values:   {BS_VALUES}")
    print(f"  max_tokens:  {MAX_OUTPUT_TOKENS} (natural EOS expected)")
    print(f"  TP:          {TP}")
    print("=" * 72)

    print("\n[mbpp] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    all_prompts, all_isls = load_mbpp_prompts(tokenizer, N_PROMPTS)
    if len(all_prompts) < N_PROMPTS:
        print(f"  Only {len(all_prompts)} prompts available in MBPP "
              f"(wanted {N_PROMPTS}); using what we have.")
    print(f"  Loaded {len(all_prompts)} prompts")
    print(f"  ISL range:   [{min(all_isls)}, {max(all_isls)}]  "
          f"mean={sum(all_isls)/len(all_isls):.0f}")
    print(f"  Total ISL:   {sum(all_isls):,}")

    bs_values = [bs for bs in BS_VALUES if bs <= len(all_prompts)]

    # Engine sizing: MBPP ISLs are very short, so this runs comfortably
    # on TP=2 even at BS=128 in a single prefill pass.
    max_model_len = 2048  # ample: longest ISL + 512 OSL fits easily
    longest_isl = max(all_isls)
    if longest_isl + MAX_OUTPUT_TOKENS >= max_model_len:
        max_model_len = longest_isl + MAX_OUTPUT_TOKENS + 64

    worst_group_total_isl = 0
    for bs in bs_values:
        n_groups = len(all_prompts) // bs
        for g in range(n_groups):
            start = g * bs
            group_total = sum(all_isls[start:start + bs])
            if group_total > worst_group_total_isl:
                worst_group_total_isl = group_total
    max_num_batched = max(worst_group_total_isl + 1, max_model_len)

    print(f"\n[mbpp] worst_group_total_isl: {worst_group_total_isl:,}")
    print(f"[mbpp] Building engine (TP={TP}, "
          f"max_num_batched={max_num_batched}, max_model_len={max_model_len})")
    t0 = time.monotonic()
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="float16",
        max_num_seqs=max(bs_values),
        max_num_batched_tokens=max_num_batched,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=42,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    ))
    print(f"[mbpp] Engine ready in {time.monotonic() - t0:.1f}s\n")

    summary_rows = []
    total_groups = sum(len(all_prompts) // bs for bs in bs_values)
    group_counter = 0

    for bs in bs_values:
        n_groups = len(all_prompts) // bs
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

            osls = [t.osl_generated for t in trackers.values()]
            reasons = [t.finish_reason for t in trackers.values()]
            n_eos = sum(1 for r in reasons if r == "stop")
            n_len = sum(1 for r in reasons if r == "length")
            prefill_ms = per_step[0]["step_ms"] if per_step else 0.0
            decode_states = {
                s["n_decoding"] for s in per_step[1:]
                if s["n_decoding"] > 0
            }
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
                "n_eos": n_eos,
                "n_length": n_len,
                "n_distinct_bs_states": n_distinct_bs_states,
            })

            if (g + 1) % max(1, n_groups // 4) == 0 or g == n_groups - 1:
                r = summary_rows[-1]
                print(f"  [{group_counter}/{total_groups}] "
                      f"bs={bs} group={g+1}/{n_groups}  "
                      f"iters={r['n_iters']:>5}  "
                      f"osl=[{r['min_osl']:>4},{r['max_osl']:>4}]  "
                      f"eos={r['n_eos']} len={r['n_length']}  "
                      f"states={r['n_distinct_bs_states']}")

        bs_elapsed = time.monotonic() - bs_t0
        print(f"  done in {bs_elapsed:.1f}s")

    summary_csv = OUT_ROOT / "drain_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bs", "group_idx", "n_requests", "total_isl", "n_iters",
            "prefill_ms", "min_osl", "max_osl", "mean_osl",
            "n_eos", "n_length", "n_distinct_bs_states",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[mbpp] Wrote {summary_csv}")
    print(f"[mbpp] {group_counter} groups done")

    print("\n[mbpp] Consolidating per-group CSVs into per-BS files...")
    consolidate(OUT_ROOT)

    print("\nPer-BS aggregate:")
    print(f"  {'BS':>4} {'groups':>7} {'avg_iters':>11} {'avg_mean_osl':>14} "
          f"{'%eos':>8} {'avg_states':>12}")
    for bs in bs_values:
        rows = [r for r in summary_rows if r["bs"] == bs]
        if not rows:
            continue
        avg_iters = sum(r["n_iters"] for r in rows) / len(rows)
        avg_mean_osl = sum(r["mean_osl"] for r in rows) / len(rows)
        total_reqs = sum(r["n_requests"] for r in rows)
        total_eos = sum(r["n_eos"] for r in rows)
        pct_eos = 100 * total_eos / max(total_reqs, 1)
        avg_states = sum(r["n_distinct_bs_states"] for r in rows) / len(rows)
        print(f"  {bs:>4} {len(rows):>7} {avg_iters:>11.0f} "
              f"{avg_mean_osl:>14.0f} {pct_eos:>7.1f}% {avg_states:>12.1f}")


if __name__ == "__main__":
    main()

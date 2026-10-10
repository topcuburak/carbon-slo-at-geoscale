#!/usr/bin/env python3
"""HumanEval TPOT drain sweep — 128 prompts × BS={1,2,4,8,16}.

HumanEval is Python function completion. Each request is a short
function signature + docstring, and the model generates a function
body until natural EOS. ISLs are small (~100-500 tokens) and OSLs
are naturally varied (~50-300 tokens), which means we should
actually see drain dynamics — short-OSL requests finish while
long-OSL requests continue decoding.

Uses the same BS sweep structure as run_tpot_drain.py and the
same (grouped) 128 prompts so results are comparable.

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=2,3 PILOT_TP=2 \\
        python datacenter/perf_models/lookup/campaign/run_humaneval_drain.py
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

DATASET = "openai_humaneval"
SPLIT = "test"
# Use all available HumanEval problems (164 total).
# Groups with non-divisible BS truncate the tail (n_groups = N_PROMPTS // bs).
N_PROMPTS = 164
BS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
# 512 is well above HumanEval's typical solution length (usually < 300)
# so natural EOS should dominate without wasted compute on outliers.
MAX_OUTPUT_TOKENS = 512
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/humaneval_drain")


def load_humaneval_prompts(tokenizer, n: int):
    """Load HumanEval problems, wrap each as an instruction prompt.

    HumanEval dataset has columns: task_id, prompt, canonical_solution,
    test, entry_point. The "prompt" field is a Python function
    signature + docstring. We wrap it in the model's chat template
    so LLaMA-3.1-Instruct responds with a completion.
    """
    ds = load_dataset(DATASET, split=SPLIT)
    prompts = []
    isls = []
    for row in ds:
        task_prompt = row["prompt"]
        messages = [
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. Only output the "
                    "function body (do not repeat the signature).\n\n"
                    f"{task_prompt}"
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
    print("  HumanEval TPOT drain sweep")
    print("=" * 72)
    print(f"  Dataset:     {DATASET} ({SPLIT})")
    print(f"  N prompts:   {N_PROMPTS}")
    print(f"  BS values:   {BS_VALUES}")
    print(f"  max_tokens:  {MAX_OUTPUT_TOKENS} (natural EOS expected)")
    print(f"  TP:          {TP}")
    print("=" * 72)

    # Load tokenizer for prompt prep
    print("\n[humaneval] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    all_prompts, all_isls = load_humaneval_prompts(tokenizer, N_PROMPTS)
    if len(all_prompts) < N_PROMPTS:
        print(f"  Only {len(all_prompts)} prompts available in HumanEval "
              f"(wanted {N_PROMPTS}); using what we have.")
    print(f"  Loaded {len(all_prompts)} prompts")
    print(f"  ISL range:   [{min(all_isls)}, {max(all_isls)}]  "
          f"mean={sum(all_isls)/len(all_isls):.0f}")
    print(f"  Total ISL:   {sum(all_isls):,}")

    # If we have fewer than 128, restrict BS values accordingly
    bs_values = [bs for bs in BS_VALUES if bs <= len(all_prompts)]

    # Engine sizing: HumanEval ISLs are short (<1K each), so this runs
    # very comfortably on TP=2 even with all 128 in one batch.
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

    print(f"\n[humaneval] worst_group_total_isl: {worst_group_total_isl:,}")
    print(f"[humaneval] Building engine (TP={TP}, "
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
    print(f"[humaneval] Engine ready in {time.monotonic() - t0:.1f}s\n")

    # ── sweep ──
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

    # ── summary CSV ──
    summary_csv = OUT_ROOT / "drain_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bs", "group_idx", "n_requests", "total_isl", "n_iters",
            "prefill_ms", "min_osl", "max_osl", "mean_osl",
            "n_eos", "n_length", "n_distinct_bs_states",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[humaneval] Wrote {summary_csv}")
    print(f"[humaneval] {group_counter} groups done")

    # Consolidate per-group CSVs into per-BS CSVs
    print("\n[humaneval] Consolidating per-group CSVs into per-BS files...")
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

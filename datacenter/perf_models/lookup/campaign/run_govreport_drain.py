#!/usr/bin/env python3
"""GovReport TPOT drain sweep — 500 samples x BS={1,2,4,8,16}.

GovReport is long-form summarization. Reports are ~2K-12K tokens,
so this sweep fills in the high-KV region of the (BS, total_KV)
lookup space that HumanEval and MBPP don't cover. We already know
LLaMA-3.1-8B-Instruct rarely reaches natural EOS on GovReport at
temperature=0, so most requests will decode up to max_tokens — this
is fine for lookup coverage: steady-state decode at constant BS with
growing KV is exactly the regime long-context workloads query.

BS is capped at 16 because a BS=32 group at ~6K avg ISL needs
~192K prefill tokens, which is tight on TP=2 even with
enable_chunked_prefill=False.

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=2,3 PILOT_TP=2 \\
        python datacenter/perf_models/lookup/campaign/run_govreport_drain.py
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

DATASET = "ccdv/govreport-summarization"
SPLIT = "test"
N_PROMPTS = 500
BS_VALUES = [1, 2, 4, 8, 16]
# GovReport doesn't naturally EOS at temperature=0; most requests will
# hit this cap. 1024 is enough to expose the long-KV decode regime while
# keeping the sweep under ~6 hours. BS capped at 16 for TP=2 memory.
MAX_OUTPUT_TOKENS = 1024
# Longest GovReport reports are ~12K tokens; 16K fits input + 1K output
# + headroom.
MAX_MODEL_LEN = 16384
TP = int(os.environ.get("PILOT_TP", "4"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUT_ROOT = Path("datacenter/perf_models/lookup/measured/archives/unpacked/govreport_drain")


def load_govreport_prompts(tokenizer, n: int):
    """Load GovReport reports, wrap each as a summarization instruction.

    GovReport has columns: report, summary. The report is a long
    gov/legal document; we wrap it in a chat template asking for a
    summary. We truncate the report itself if it would push past
    MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - overhead, so every prompt
    safely fits the engine.
    """
    ds = load_dataset(DATASET, split=SPLIT)
    prompts = []
    isls = []
    # Reserve room for output + chat template tokens.
    max_prompt_tokens = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - 128
    for row in ds:
        report = row["report"]
        # Pre-truncate by tokens to avoid length-cap failures at runtime.
        report_ids = tokenizer.encode(report, add_special_tokens=False)
        if len(report_ids) > max_prompt_tokens - 128:
            report_ids = report_ids[: max_prompt_tokens - 128]
            report = tokenizer.decode(report_ids, skip_special_tokens=True)
        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following government report in a few "
                    "concise paragraphs.\n\n"
                    f"{report}"
                ),
            },
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tok_len = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
        if tok_len > max_prompt_tokens:
            continue
        prompts.append(chat_prompt)
        isls.append(tok_len)
        if len(prompts) >= n:
            break
    return prompts, isls


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  GovReport TPOT drain sweep")
    print("=" * 72)
    print(f"  Dataset:     {DATASET} ({SPLIT})")
    print(f"  N prompts:   {N_PROMPTS}")
    print(f"  BS values:   {BS_VALUES}")
    print(f"  max_tokens:  {MAX_OUTPUT_TOKENS} (cap expected to dominate)")
    print(f"  TP:          {TP}")
    print("=" * 72)

    print("\n[govreport] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    all_prompts, all_isls = load_govreport_prompts(tokenizer, N_PROMPTS)
    if len(all_prompts) < N_PROMPTS:
        print(f"  Only {len(all_prompts)} prompts available after filtering "
              f"(wanted {N_PROMPTS}); using what we have.")
    print(f"  Loaded {len(all_prompts)} prompts")
    print(f"  ISL range:   [{min(all_isls)}, {max(all_isls)}]  "
          f"mean={sum(all_isls)/len(all_isls):.0f}")
    print(f"  Total ISL:   {sum(all_isls):,}")

    bs_values = [bs for bs in BS_VALUES if bs <= len(all_prompts)]

    # Engine sizing: max_num_batched must cover the worst BS x ISL group.
    max_model_len = MAX_MODEL_LEN
    worst_group_total_isl = 0
    for bs in bs_values:
        n_groups = len(all_prompts) // bs
        for g in range(n_groups):
            start = g * bs
            group_total = sum(all_isls[start:start + bs])
            if group_total > worst_group_total_isl:
                worst_group_total_isl = group_total
    max_num_batched = max(worst_group_total_isl + 1, max_model_len)

    print(f"\n[govreport] worst_group_total_isl: {worst_group_total_isl:,}")
    print(f"[govreport] Building engine (TP={TP}, "
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
    print(f"[govreport] Engine ready in {time.monotonic() - t0:.1f}s\n")

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

    print(f"\n[govreport] Wrote {summary_csv}")
    print(f"[govreport] {group_counter} groups done")

    print("\n[govreport] Consolidating per-group CSVs into per-BS files...")
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

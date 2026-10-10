#!/usr/bin/env python3
"""LLaMA-3.1-70B TPOT drain sweep for HumanEval / MBPP / GovReport.

Unified script that runs the same batch-drain protocol as the
dataset-specific 8B scripts (run_humaneval_drain.py, run_mbpp_drain.py,
run_govreport_drain.py) but on LLaMA-3.1-70B-Instruct at TP=4.

The 70B model is ~140 GB in FP16, so TP=4 on 4× A100 is the minimum
viable setup. KV cache budget per GPU is tight (A100-40GB: ~5 GB left
after model weights), so BS values and max_model_len are capped
conservatively relative to the 8B sweeps.

Outputs are written under datacenter/perf_models/lookup/measured/archives/unpacked/<dataset>_drain_70b/
to sit alongside the 8B results for easy side-by-side comparison.

Usage:
    conda activate llm_bench
    CUDA_VISIBLE_DEVICES=0,1,2,3 PILOT_TP=4 \\
        python datacenter/perf_models/lookup/campaign/run_drain_70b.py --dataset humaneval
    CUDA_VISIBLE_DEVICES=0,1,2,3 PILOT_TP=4 \\
        python datacenter/perf_models/lookup/campaign/run_drain_70b.py --dataset mbpp
    CUDA_VISIBLE_DEVICES=0,1,2,3 PILOT_TP=4 \\
        python datacenter/perf_models/lookup/campaign/run_drain_70b.py --dataset govreport

Arguments:
    --dataset {humaneval,mbpp,govreport}  Which dataset to sweep.
    --max-model-len INT                    Override per-sequence context.
    --max-num-seqs INT                     Override max concurrent seqs.
    --bs-values CSV                        Override BS values (e.g. "1,2,4,8").
    --n-prompts INT                        Override prompt count.
"""
from __future__ import annotations

import argparse
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


# ── global config ───────────────────────────────────────────────────────

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
TP = int(os.environ.get("PILOT_TP", "4"))

# Per-dataset defaults. Each entry:
#   hf_key: HF dataset loader key (string) or (repo, config) tuple for MBPP
#   split: HF split to use
#   n_prompts: how many to sample
#   bs_values: batch-size sweep (capped conservatively for 70B memory)
#   max_output_tokens: decode cap
#   max_model_len: per-sequence context limit
#   out_dir: relative path under datacenter/perf_models/lookup/measured/archives/unpacked/
DATASET_CONFIGS = {
    "humaneval": {
        "hf_key": "openai_humaneval",
        "split": "test",
        "n_prompts": 164,
        "bs_values": [1, 2, 4, 8, 16, 32],
        "max_output_tokens": 512,
        "max_model_len": 2048,
        "out_dir": Path("datacenter/perf_models/lookup/measured/archives/unpacked/humaneval_drain_70b"),
        "prompt_field": "prompt",
        "instruction": (
            "Complete the following Python function. Only output the "
            "function body (do not repeat the signature)."
        ),
        "include_test_hint": False,
    },
    "mbpp": {
        "hf_key": ("google-research-datasets/mbpp", "full"),
        "splits": ["train", "test", "validation", "prompt"],
        "n_prompts": 974,
        "bs_values": [1, 2, 4, 8, 16, 32],
        "max_output_tokens": 512,
        "max_model_len": 2048,
        "out_dir": Path("datacenter/perf_models/lookup/measured/archives/unpacked/mbpp_drain_70b"),
        "prompt_field": "text",
        "instruction": (
            "Write a Python function for the following task. Only output "
            "the function definition."
        ),
        "include_test_hint": True,
    },
    "govreport": {
        "hf_key": "ccdv/govreport-summarization",
        "split": "test",
        "n_prompts": 500,
        "bs_values": [1, 2, 4, 8, 16, 32],
        "max_output_tokens": 1024,
        "max_model_len": 16384,
        "out_dir": Path("datacenter/perf_models/lookup/measured/archives/unpacked/govreport_drain_70b"),
        "prompt_field": "report",
        "instruction": (
            "Summarize the following government report in a few concise "
            "paragraphs."
        ),
        "include_test_hint": False,
    },
}


# ── dataset loaders ─────────────────────────────────────────────────────

def _wrap_chat(tokenizer, instruction, body):
    messages = [{"role": "user", "content": f"{instruction}\n\n{body}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def load_humaneval(tokenizer, n, max_prompt_tokens, cfg):
    ds = load_dataset(cfg["hf_key"], split=cfg["split"])
    prompts, isls = [], []
    for row in ds:
        task_prompt = row[cfg["prompt_field"]]
        chat = _wrap_chat(tokenizer, cfg["instruction"], task_prompt)
        tok_len = len(tokenizer.encode(chat, add_special_tokens=False))
        prompts.append(chat)
        isls.append(tok_len)
        if len(prompts) >= n:
            break
    return prompts, isls


def load_mbpp(tokenizer, n, max_prompt_tokens, cfg):
    hf_path, hf_config = cfg["hf_key"]
    rows = []
    for split in cfg["splits"]:
        ds = load_dataset(hf_path, hf_config, split=split)
        for row in ds:
            rows.append(row)
            if len(rows) >= n:
                break
        if len(rows) >= n:
            break

    prompts, isls = [], []
    for row in rows:
        task_text = row[cfg["prompt_field"]]
        hint = ""
        if cfg["include_test_hint"]:
            tests = row.get("test_list") or []
            if tests:
                hint = f"\n\nYour function must pass this test:\n{tests[0]}"
        chat = _wrap_chat(tokenizer, cfg["instruction"], f"{task_text}{hint}")
        tok_len = len(tokenizer.encode(chat, add_special_tokens=False))
        prompts.append(chat)
        isls.append(tok_len)
        if len(prompts) >= n:
            break
    return prompts, isls


def load_govreport(tokenizer, n, max_prompt_tokens, cfg):
    ds = load_dataset(cfg["hf_key"], split=cfg["split"])
    prompts, isls = [], []
    for row in ds:
        report = row[cfg["prompt_field"]]
        # Pre-truncate by tokens so every prompt fits in max_prompt_tokens.
        report_ids = tokenizer.encode(report, add_special_tokens=False)
        if len(report_ids) > max_prompt_tokens - 128:
            report_ids = report_ids[: max_prompt_tokens - 128]
            report = tokenizer.decode(report_ids, skip_special_tokens=True)
        chat = _wrap_chat(tokenizer, cfg["instruction"], report)
        tok_len = len(tokenizer.encode(chat, add_special_tokens=False))
        if tok_len > max_prompt_tokens:
            continue
        prompts.append(chat)
        isls.append(tok_len)
        if len(prompts) >= n:
            break
    return prompts, isls


LOADERS = {
    "humaneval": load_humaneval,
    "mbpp":      load_mbpp,
    "govreport": load_govreport,
}


# ── main driver ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLaMA-70B drain sweep")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override per-sequence context limit.")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Override vLLM max concurrent sequences.")
    parser.add_argument("--bs-values", type=str, default=None,
                        help="Comma-separated BS values (e.g. '1,2,4,8').")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Override number of prompts to sample.")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    bs_values = [int(b) for b in args.bs_values.split(",")] if args.bs_values \
                else cfg["bs_values"]
    max_model_len = args.max_model_len or cfg["max_model_len"]
    max_output_tokens = cfg["max_output_tokens"]
    n_prompts = args.n_prompts or cfg["n_prompts"]
    out_root = cfg["out_dir"]

    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  LLaMA-70B TPOT drain sweep — {args.dataset}")
    print("=" * 72)
    print(f"  Model:         {MODEL}")
    print(f"  TP:            {TP}")
    print(f"  N prompts:     {n_prompts}")
    print(f"  BS values:     {bs_values}")
    print(f"  max_tokens:    {max_output_tokens}")
    print(f"  max_model_len: {max_model_len}")
    print(f"  out dir:       {out_root}")
    print("=" * 72)

    # ── load prompts ──
    print(f"\n[{args.dataset}-70b] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    max_prompt_tokens = max_model_len - max_output_tokens - 128
    loader = LOADERS[args.dataset]
    all_prompts, all_isls = loader(tokenizer, n_prompts, max_prompt_tokens, cfg)
    print(f"  Loaded {len(all_prompts)} prompts")
    print(f"  ISL range: [{min(all_isls)}, {max(all_isls)}]  "
          f"mean={sum(all_isls)/len(all_isls):.0f}")
    print(f"  Total ISL: {sum(all_isls):,}")

    bs_values = [bs for bs in bs_values if bs <= len(all_prompts)]

    # ── engine sizing ──
    longest_isl = max(all_isls)
    if longest_isl + max_output_tokens >= max_model_len:
        max_model_len = longest_isl + max_output_tokens + 64
        print(f"  Bumped max_model_len to {max_model_len} to fit longest ISL")

    worst_group_total_isl = 0
    for bs in bs_values:
        n_groups = len(all_prompts) // bs
        for g in range(n_groups):
            start = g * bs
            worst_group_total_isl = max(
                worst_group_total_isl,
                sum(all_isls[start:start + bs])
            )
    max_num_batched = max(worst_group_total_isl + 1, max_model_len)

    max_num_seqs_arg = args.max_num_seqs or max(bs_values)

    print(f"\n[{args.dataset}-70b] worst_group_total_isl: {worst_group_total_isl:,}")
    print(f"[{args.dataset}-70b] Building engine "
          f"(TP={TP}, max_num_batched={max_num_batched}, "
          f"max_model_len={max_model_len}, max_num_seqs={max_num_seqs_arg})")
    t0 = time.monotonic()
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="float16",
        max_num_seqs=max_num_seqs_arg,
        max_num_batched_tokens=max_num_batched,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=42,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    ))
    print(f"[{args.dataset}-70b] Engine ready in {time.monotonic() - t0:.1f}s\n")

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
                out_dir=out_root / batch_id,
                model=MODEL,
                max_output_tokens=max_output_tokens,
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

        print(f"  done in {time.monotonic() - bs_t0:.1f}s")

    summary_csv = out_root / "drain_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bs", "group_idx", "n_requests", "total_isl", "n_iters",
            "prefill_ms", "min_osl", "max_osl", "mean_osl",
            "n_eos", "n_length", "n_distinct_bs_states",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[{args.dataset}-70b] Wrote {summary_csv}")
    print(f"[{args.dataset}-70b] {group_counter} groups done")

    print(f"\n[{args.dataset}-70b] Consolidating per-group CSVs into per-BS files...")
    consolidate(out_root)

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

#!/usr/bin/env python3
"""Run one homogeneous batch with per-engine-step timing.

Uses vLLM's low-level LLMEngine so we can time every `engine.step()` call
and log (iteration, n_prefilling, n_decoding, step_ms, total_kv, active_ids).

Configuration:
    - enable_chunked_prefill = False       (forces single-pass prefill)
    - max_num_batched_tokens  = auto        (set large enough for the batch's
                                             total ISL so all prefill happens
                                             in one step)
    - max_num_seqs           = batch_size   (exact batch)
    - tensor_parallel_size   = 4
    - enforce_eager          = True         (skip CUDA graphs — cleaner timing)
    - dtype                  = float16

Outputs:
    results/<batch_id>/per_step.csv
        iter, step_type, n_prefilling, n_decoding, n_active,
        step_ms, total_kv, prefill_tokens, decode_tokens, active_ids

    results/<batch_id>/per_request.csv
        req_id, isl, osl_requested, osl_generated,
        ttft_ms, decode_total_ms, decode_steps,
        arrival_iter, first_token_iter, final_iter

    results/<batch_id>/config.json
        batch_id, dataset, bs, max_output_tokens, vllm_version, ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.inputs import TextPrompt


# ────────────────────────── schema definitions ──────────────────────────

PER_STEP_FIELDS = [
    "iter",
    "step_type",          # "prefill" / "decode" / "mixed" / "idle"
    "n_prefilling",       # requests in prefill phase this step
    "n_decoding",         # requests generating tokens this step
    "n_active",           # total alive (prefilling + decoding)
    "step_ms",            # measured wall time of engine.step()
    "total_kv_before",    # sum of (context_len) before this step for all active seqs
    "total_kv_after",     # sum of (context_len) after this step
    "prefill_tokens",     # prefill tokens processed this step
    "decode_tokens",      # decode tokens generated this step (= n_decoding in pure decode)
    "active_ids",         # pipe-separated request ids present in this step
]

PER_REQUEST_FIELDS = [
    "req_id",
    "isl",                # input tokens
    "osl_requested",      # max_new_tokens sent
    "osl_generated",      # actual output length
    "ttft_ms",            # ms from add_request to first generated token
    "decode_total_ms",    # ms from first to last generated token
    "decode_steps",       # number of decode steps this request participated in
    "arrival_iter",       # iteration when request was added
    "first_token_iter",   # iteration when it emitted its first decode token
    "final_iter",         # iteration when it finished
    "finish_reason",
]


# ────────────────────────── per-request tracker ──────────────────────────

@dataclass
class RequestTracker:
    req_id: str
    isl: int
    osl_requested: int
    arrival_time_ms: float
    arrival_iter: int
    osl_generated: int = 0
    first_token_time_ms: Optional[float] = None
    first_token_iter: Optional[int] = None
    final_time_ms: Optional[float] = None
    final_iter: Optional[int] = None
    decode_steps: int = 0
    finish_reason: str = ""


# ────────────────────────── main runner ──────────────────────────

def run_batch(
    prompts: List[str],
    isls: List[int],
    batch_id: str,
    out_dir: Path,
    model: str,
    max_output_tokens: int,
    tensor_parallel_size: int = 4,
    dtype: str = "float16",
    seed: int = 42,
    engine: Optional[LLMEngine] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, RequestTracker], List[dict]]:
    """Run a single homogeneous batch, return per-request trackers + per-step logs.

    If `engine` is provided, it is reused (useful for the pilot that runs
    multiple batches back-to-back without reloading the model).
    """
    assert len(prompts) == len(isls), "prompts and isls must match"
    n = len(prompts)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_isl = sum(isls)
    # max_num_batched_tokens must cover total prefill in one pass
    max_num_batched = max(total_isl + 1, 32768)

    if engine is None:
        # Fixed max_model_len=64K per-sequence context ceiling.
        # vLLM requires max_num_batched_tokens >= max_model_len; also
        # must cover the batch's total prefill in one pass.
        max_model_len = 65536
        max_num_batched = max(max_num_batched, max_model_len)

        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_num_seqs=max(n, 1),
            max_num_batched_tokens=max_num_batched,
            max_model_len=max_model_len,
            enable_chunked_prefill=False,
            enforce_eager=True,
            seed=seed,
            gpu_memory_utilization=0.90,
            disable_log_stats=True,
        )
        if verbose:
            print(f"[{batch_id}] Creating engine "
                  f"(TP={tensor_parallel_size}, max_batch_tokens={max_num_batched})...")
        engine = LLMEngine.from_engine_args(engine_args)

    # Stop strings catch common runaway-repetition patterns (e.g. the model
    # stuck printing blank lines or punctuation sequences). Not foolproof
    # but avoids pathological 16K-token "loops" that block the sweep.
    stop_strings = ["\n\n\n\n\n", "....................", "--------------------"]
    # If max_output_tokens is 0 or negative, don't cap — let vLLM generate
    # until natural EOS (limited only by max_model_len).
    if max_output_tokens and max_output_tokens > 0:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_output_tokens,
            ignore_eos=False,
            stop=stop_strings,
        )
    else:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=None,
            ignore_eos=False,
            stop=stop_strings,
        )

    # Add all requests at once — all prefill happens in the first step
    trackers: Dict[str, RequestTracker] = {}
    arrival_time_ms = time.monotonic() * 1000.0
    for i, (prompt, isl) in enumerate(zip(prompts, isls)):
        rid = f"{batch_id}_req{i}"
        engine.add_request(
            request_id=rid,
            prompt=TextPrompt(prompt=prompt),
            params=sampling,
        )
        trackers[rid] = RequestTracker(
            req_id=rid,
            isl=isl,
            osl_requested=max_output_tokens,
            arrival_time_ms=arrival_time_ms,
            arrival_iter=0,
        )

    if verbose:
        print(f"[{batch_id}] Added {n} requests "
              f"(total_ISL={total_isl}, max_output_tokens={max_output_tokens})")

    # Drive engine.step() in a loop, timing each call
    per_step_logs: List[dict] = []
    iter_idx = 0
    finished_ids: set = set()

    t_start_wall = time.monotonic()

    while engine.has_unfinished_requests():
        # Snapshot state BEFORE the step
        before = _snapshot_engine(engine)

        t_step = time.monotonic()
        request_outputs = engine.step()
        step_ms = (time.monotonic() - t_step) * 1000.0

        after = _snapshot_engine(engine)

        # Figure out step type
        step_type = _classify_step(before, after)

        # Build per-step log row
        active_ids_before = sorted(before["running_ids"] | before["waiting_ids"])
        prefill_tokens = after["total_prompt_tokens_processed"] - before["total_prompt_tokens_processed"]
        decode_tokens = len([
            ro for ro in request_outputs
            if ro.outputs and len(ro.outputs[0].token_ids) > 0
        ])

        per_step_logs.append({
            "iter": iter_idx,
            "step_type": step_type,
            "n_prefilling": before["n_prefilling"],
            "n_decoding": before["n_decoding"],
            "n_active": before["n_prefilling"] + before["n_decoding"],
            "step_ms": round(step_ms, 4),
            "total_kv_before": before["total_context_tokens"],
            "total_kv_after": after["total_context_tokens"],
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "active_ids": "|".join(active_ids_before),
        })

        # Update per-request state from the step's outputs
        now_ms = time.monotonic() * 1000.0
        for ro in request_outputs:
            rid = ro.request_id
            if rid not in trackers:
                continue
            tracker = trackers[rid]
            out = ro.outputs[0] if ro.outputs else None
            n_tokens = len(out.token_ids) if out else 0

            # First token?
            if tracker.first_token_time_ms is None and n_tokens > 0:
                tracker.first_token_time_ms = now_ms
                tracker.first_token_iter = iter_idx

            # Count decode step if this request generated something
            if n_tokens > tracker.osl_generated:
                tracker.decode_steps += 1
                tracker.osl_generated = n_tokens

            # Finished?
            if ro.finished and rid not in finished_ids:
                tracker.final_time_ms = now_ms
                tracker.final_iter = iter_idx
                tracker.finish_reason = out.finish_reason if out else "unknown"
                finished_ids.add(rid)

        iter_idx += 1

    total_wall_ms = (time.monotonic() - t_start_wall) * 1000.0

    if verbose:
        print(f"[{batch_id}] Finished in {total_wall_ms/1000:.1f}s "
              f"({iter_idx} iterations, {len(finished_ids)}/{n} requests)")

    # ── Write outputs ──
    _write_per_step(out_dir / "per_step.csv", per_step_logs)
    _write_per_request(out_dir / "per_request.csv", trackers)

    config = {
        "batch_id": batch_id,
        "n_requests": n,
        "total_isl": total_isl,
        "max_output_tokens": max_output_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "max_num_batched_tokens": max_num_batched,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "total_wall_ms": round(total_wall_ms, 2),
        "total_iterations": iter_idx,
        "model": model,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return trackers, per_step_logs


def _snapshot_engine(engine: LLMEngine) -> dict:
    """Extract current engine state as a snapshot dict.

    vLLM's LLMEngine exposes:
      engine.scheduler  → one per pipeline stage; we read [0]
      scheduler.running  → list of SequenceGroups currently in running (usually decoding)
      scheduler.waiting  → list in waiting (prefill queue)
      scheduler.swapped  → list in swapped (paged out)
    """
    scheduler = engine.scheduler[0] if isinstance(engine.scheduler, list) else engine.scheduler

    running = list(scheduler.running)
    waiting = list(scheduler.waiting)
    swapped = list(scheduler.swapped) if hasattr(scheduler, "swapped") else []

    # Classify each running SG as prefilling or decoding
    n_prefilling = 0
    n_decoding = 0
    total_context = 0
    total_prompt_processed = 0

    for sg in running:
        for seq in sg.get_seqs():
            n_tokens = seq.get_num_computed_tokens() if hasattr(seq, 'get_num_computed_tokens') else 0
            prompt_len = seq.get_prompt_len() if hasattr(seq, 'get_prompt_len') else len(seq.prompt_token_ids)
            output_len = seq.get_output_len() if hasattr(seq, 'get_output_len') else 0
            total_context += prompt_len + output_len
            total_prompt_processed += min(n_tokens, prompt_len)
            # If still in prefill, n_computed < prompt_len
            if hasattr(seq, 'get_num_computed_tokens'):
                if seq.get_num_computed_tokens() < prompt_len:
                    n_prefilling += 1
                else:
                    n_decoding += 1
            else:
                # Fallback: assume decoding
                n_decoding += 1

    running_ids = set()
    for sg in running:
        running_ids.add(sg.request_id)
    waiting_ids = set()
    for sg in waiting:
        waiting_ids.add(sg.request_id)

    return {
        "n_running": len(running),
        "n_waiting": len(waiting),
        "n_swapped": len(swapped),
        "n_prefilling": n_prefilling,
        "n_decoding": n_decoding,
        "total_context_tokens": total_context,
        "total_prompt_tokens_processed": total_prompt_processed,
        "running_ids": running_ids,
        "waiting_ids": waiting_ids,
    }


def _classify_step(before: dict, after: dict) -> str:
    pre = before["n_prefilling"]
    dec = before["n_decoding"]
    if pre > 0 and dec > 0:
        return "mixed"
    if pre > 0:
        return "prefill"
    if dec > 0:
        return "decode"
    return "idle"


def _write_per_step(path: Path, rows: List[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_STEP_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PER_STEP_FIELDS})


def _write_per_request(path: Path, trackers: Dict[str, RequestTracker]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_REQUEST_FIELDS)
        w.writeheader()
        for t in trackers.values():
            ttft_ms = ""
            if t.first_token_time_ms is not None:
                ttft_ms = round(t.first_token_time_ms - t.arrival_time_ms, 3)
            decode_total_ms = ""
            if (t.first_token_time_ms is not None
                    and t.final_time_ms is not None):
                decode_total_ms = round(
                    t.final_time_ms - t.first_token_time_ms, 3)
            w.writerow({
                "req_id": t.req_id,
                "isl": t.isl,
                "osl_requested": t.osl_requested,
                "osl_generated": t.osl_generated,
                "ttft_ms": ttft_ms,
                "decode_total_ms": decode_total_ms,
                "decode_steps": t.decode_steps,
                "arrival_iter": t.arrival_iter,
                "first_token_iter": (
                    t.first_token_iter
                    if t.first_token_iter is not None else ""),
                "final_iter": (
                    t.final_iter if t.final_iter is not None else ""),
                "finish_reason": t.finish_reason,
            })


# ────────────────────────── CLI ──────────────────────────

def _load_prompts_from_execlog(
    execlog_csv: str,
    n: int,
    offset: int = 0,
) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(execlog_csv)
    df = df.iloc[offset:offset + n]
    prompts = df["prompt"].astype(str).tolist()
    isls = df["in_tokens"].astype(int).tolist()
    return prompts, isls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execlog", required=True,
                    help="Path to execution_logs CSV with prompt column")
    ap.add_argument("--bs", type=int, required=True, help="Batch size")
    ap.add_argument("--offset", type=int, default=0,
                    help="Start index into execlog")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--out-dir", default="datacenter/perf_models/lookup/measured/archives/unpacked")
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--model",
                    default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    prompts, isls = _load_prompts_from_execlog(args.execlog, args.bs, args.offset)
    out_dir = Path(args.out_dir) / args.batch_id

    run_batch(
        prompts=prompts,
        isls=isls,
        batch_id=args.batch_id,
        out_dir=out_dir,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        tensor_parallel_size=args.tp,
    )


if __name__ == "__main__":
    main()

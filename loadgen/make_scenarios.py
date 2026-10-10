"""Build the inference-scenario traces the sweep replays.

Three families, each generated for every model, because ISL is a property of
the tokenizer and not of the text:

    mlperf_<model>  what MLPerf runs for that model. CNN-DailyMail 3.0.0
                    validation for llama and qwen -- the 13368-sample split the
                    rules quote -- and OpenOrca for mixtral, which is the
                    benchmark MLPerf uses there. Short: mean ISL 841 under the
                    llama tokenizer, against the 840 the testbed harness reports
                    for the same dataset and the 778 MLPerf quotes.

    mixed_<model>   every source, long prompts included. Spans the space a
                    serving fleet actually sees: code, maths, chat, medium
                    summarization, long-form summarization.

    long_<model>    GovReport only, floored at --long-min-isl. The long-context
                    regime on its own, which is where the prefill attention term
                    and KV pressure bind.

Same sources under three tokenizers, so a comparison across models is a
comparison of models. mixed and long are identical text for all three; only
mlperf differs by model, because MLPerf itself does.

OSL is the tokenized REFERENCE output -- the human summary, the dataset's own
response, the reference solution -- which is what MLPerf scores against. The
testbed harness pins max_tokens instead, which is right for a calibration block
and wrong for a workload.

ISL and OSL are capped so that ISL + OSL never exceeds --max-ctx. Past it the
engine would truncate generation and the simulator would have to clamp, and a
trace should not be built to need that.

sample_id names the SAMPLE and is deliberately not the id that goes on the
wire: the load generator cycles the trace when a run outlasts it, so the same
sample is dispatched repeatedly and a shared id would stop telling two
in-flight requests apart. The two relate without another column, because the
cycle is deterministic:

    sample_id = (req_num - 1) mod len(trace) + 1

src_id is the source dataset's own id, so a row is always traceable back.

Deterministic: the shuffle is seeded by index rather than by the datasets
library's internals, which have changed across versions. A regenerated prefix
is byte-identical to the full file's prefix, so the trace can be consumed and
regenerated identically.

    python -m loadgen.make_scenarios --families mlperf mixed long
"""

import argparse
import csv
import random
from pathlib import Path

# name -> (hf path, load kwargs, split, input fields joined by \n, output field)
SOURCES = {
    "cnn":       ("abisee/cnn_dailymail", {"name": "3.0.0"}, "validation",
                  ["article"], "highlights"),
    "govreport": ("ccdv/govreport-summarization", {}, "train",
                  ["report"], "summary"),
    "openorca":  ("ctuning/ml_perf-open_orca", {}, "train",
                  ["system_prompt", "question"], "response"),
    "gsm8k":     ("openai/gsm8k", {"name": "main"}, "train",
                  ["question"], "answer"),
    "mbpp":      ("google-research-datasets/mbpp", {}, "train",
                  ["text"], "code"),
    "humaneval": ("openai/openai_humaneval", {}, "test",
                  ["prompt"], "canonical_solution"),
}

MODELS = {
    "llama3_8B":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5_7B":   "Qwen/Qwen2.5-7B-Instruct",
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# MLPerf runs a different benchmark for mixtral than for the dense models.
MLPERF_SOURCE = {"llama3_8B": "cnn", "qwen2.5_7B": "cnn",
                 "mixtral_8x7b": "openorca"}

# Every source, capped equally so none dominates by the accident of its size.
MIXED_SOURCES = ["govreport", "cnn", "openorca", "gsm8k", "mbpp", "humaneval"]
LONG_SOURCES = ["govreport"]


def _rows_from(source, tok, max_isl, min_isl, max_osl, cap, seed, max_ctx):
    from datasets import load_dataset
    hf, kw, split, in_fields, out_field = SOURCES[source]
    ds = load_dataset(hf, split=split, **kw)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    out, skipped = [], 0
    for i in idx:
        rec = ds[i]
        text = "\n".join(str(rec[f]) for f in in_fields if rec.get(f))
        in_ids = tok(text, add_special_tokens=False)["input_ids"][:max_isl]
        if len(in_ids) < min_isl:
            skipped += 1
            continue
        # Leave the output room inside the context window rather than letting
        # the engine truncate it later.
        room = max(max_ctx - len(in_ids), 1)
        out_ids = tok(str(rec[out_field]), add_special_tokens=False)["input_ids"][:min(max_osl, room)]
        if not out_ids:
            skipped += 1
            continue
        out.append({"in_tok": len(in_ids), "out_tok": len(out_ids),
                    "dataset": source, "src_id": str(rec.get("id", rec.get("task_id", i)))})
        if cap and len(out) >= cap:
            break
    return out, skipped


def build(family, model_key, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])

    if family == "mlperf":
        sources, min_isl, max_isl = [MLPERF_SOURCE[model_key]], args.min_isl, args.mlperf_max_isl
    elif family == "mixed":
        sources, min_isl, max_isl = MIXED_SOURCES, args.min_isl, args.max_isl
    else:
        sources, min_isl, max_isl = LONG_SOURCES, args.long_min_isl, args.max_isl

    # The per-source cap exists so that no source dominates a MIX by the
    # accident of its size. A single-source family has nothing to balance, so
    # it takes the whole split -- which for mlperf is also what makes the run
    # comparable, since the rules quote the sample count.
    cap = args.per_source if len(sources) > 1 else 0

    rows = []
    for s in sources:
        got, skipped = _rows_from(s, tok, max_isl, min_isl, args.max_osl,
                                  cap, args.seed, args.max_ctx)
        print(f"    {s}: {len(got)} kept, {skipped} skipped")
        rows.extend(got)

    # One shuffle over the pool so datasets interleave rather than arriving in
    # blocks; a run shorter than the trace must still see the whole mix.
    random.Random(args.seed).shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["sample_id"] = i

    out = Path(args.out_dir) / f"{family}_{model_key}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "in_tok", "out_tok",
                                          "dataset", "src_id"])
        w.writeheader()
        w.writerows(rows)

    ins = sorted(r["in_tok"] for r in rows)
    outs = sorted(r["out_tok"] for r in rows)
    pi = sum(ins); po = sum(outs)
    q = lambda a, p: a[min(int(p * len(a)), len(a) - 1)]
    print(f"  {out}: {len(rows)} requests")
    print(f"    ISL p50 {q(ins,.5)} p90 {q(ins,.9)} max {ins[-1]} mean {pi//len(rows)} | "
          f"OSL p50 {q(outs,.5)} p90 {q(outs,.9)} max {outs[-1]} | "
          f"prefill {100*pi/(pi+po):.0f}% / decode {100*po/(pi+po):.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["mlperf", "mixed", "long"],
                    choices=["mlperf", "mixed", "long"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out-dir", default="loadgen/scenarios")
    ap.add_argument("--mlperf-max-isl", type=int, default=2048,
                    help="MLPerf's max_seq_len for these benchmarks")
    ap.add_argument("--max-isl", type=int, default=30720,
                    help="mixed and long; leaves output room inside --max-ctx")
    ap.add_argument("--max-ctx", type=int, default=32768,
                    help="ISL+OSL ceiling; the sweep's MAX_MODEL_LEN and where "
                         "the calibration envelope stops")
    ap.add_argument("--min-isl", type=int, default=16)
    ap.add_argument("--long-min-isl", type=int, default=2048,
                    help="what counts as a long prompt")
    ap.add_argument("--max-osl", type=int, default=2048)
    ap.add_argument("--per-source", type=int, default=4000,
                    help="cap per source, so none dominates by its size alone")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for family in args.families:
        for m in args.models:
            print(f"[{family}] {m}")
            build(family, m, args)


if __name__ == "__main__":
    main()

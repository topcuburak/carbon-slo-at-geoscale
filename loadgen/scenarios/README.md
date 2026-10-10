# Scenario traces

Every trace the load generator replays lives here. Two generations.

## Current: `<family>_<model>.csv`

Built by `python -m loadgen.make_scenarios`. Three families for each of
llama3_8B, qwen2.5_7B and mixtral_8x7b:

| family | sources | regime |
|---|---|---|
| `mlperf` | CNN-DailyMail 3.0.0 validation (llama, qwen); OpenOrca (mixtral) | what MLPerf runs for that model |
| `mixed` | GovReport, CNN, OpenOrca, GSM8K, MBPP, HumanEval | everything, long prompts included |
| `long` | GovReport, ISL floored at 2048 | long context on its own |

`mixed` and `long` are the same text for all three models, so a comparison
across models is a comparison of models rather than of workloads; only `mlperf`
draws on a different source per model, because MLPerf itself does. Each source
is capped equally (`--per-source`) so none dominates by the accident of its
size, and the pool is shuffled once so datasets interleave.

Columns: `sample_id, in_tok, out_tok, dataset, src_id`. OSL is the tokenized
reference output, not a pinned `max_tokens`. ISL+OSL never exceeds `--max-ctx`
(32768, the sweep's `MAX_MODEL_LEN` and where the calibration envelope stops),
so the simulator's context clamp never fires on these.

`sample_id` names the SAMPLE, not the dispatch. The generator cycles the trace
when a run outlasts it, so the same sample goes out repeatedly; the id on the
wire is the client's own counter, which stays unique. They relate as
`sample_id = (req_num - 1) mod len(trace) + 1`.

Deterministic: the shuffle is seeded by index, so a regenerated prefix is
byte-identical to the full file's prefix.

## Earlier: `mixed_trace*.csv`, `trace_*.csv`

Their ISL and OSL are MEASURED from real inference runs -- `out_tok` is what
the model actually generated -- rather than tokenized from the source text, so
they record what a model did where the current families define what it should
do. `mixed_trace_3ds.csv` is still the sweep's default.

The generator that built them is gone: nothing called it, and its input
(`datacenter/perf_models/lookup/measured/full/`) is not in the repo, so it could not have run. These
three files are the whole of what it produced.

Two things to know before using them. Their `sample_id` is a row index and the
generator threw the benchmark's own id away, so no two of these traces can be
joined -- which is why the `predicted_osl` on `mixed_trace_with_osl.csv` cannot
be transferred to `mixed_trace_3ds.csv`, and why the OSL sensitivity arm has to
run both ways on the former. And `mixed_trace_3ds.csv` has 2 requests whose
ISL+OSL exceeds 32768; the engine would truncate them and the simulator now
clamps them.


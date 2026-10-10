# Predicting how long an answer will be

The scheduler prices a placement on the request's output length, and it has two
ways to get one. `OSL_SOURCE=oracle` reads the true `out_tok` off the trace,
which makes it clairvoyant and is an upper bound rather than a policy.
`OSL_SOURCE=predicted` reads a `predicted_osl` column instead. This directory
is where that column comes from.

**Nothing here runs during a simulation.** The five packages that do --
datacenter, scheduler, provisioner, loadgen, common -- import nothing from
here. The predictor is applied offline, once, and its answers are written into
the scenario CSVs:

    workload_harness/     runs vLLM over a dataset and records one row per
                          request: prompt, the length the model actually chose,
                          latency
        -> results/workload/          128 files
    train.py, train_l1.py, models.py, losses.py
    data_loader.py        the per-request rows from results/workload/
    benchmark_loader.py   the same pairs out of the benchmark CSVs; L1 trains
                          on these. Named for where the CSVs come from, not for
                          what it is used for
        -> checkpoints/
    predict.py            inference-time wrapper, targeting the L2/L3
                          transformer that is not built
    fit_scenarios.py      fill predicted_osl on loadgen/scenarios/*.csv
        -> the column the scheduler reads

So you need this directory when regenerating the scenarios or refitting the
predictor, and at no other time.

## The column is cross-fitted, and that is the point

A predictor trained on all the rows and then asked about those rows reports
memorisation: `predicted_osl` approaches `out_tok`, and the OSL-sensitivity arm
concludes a predictor is as good as an oracle -- a result about the fit, not
about scheduling. `fit_scenarios.py` splits the rows into K folds and takes each
fold's predictions from a model trained on the other K-1, so every value in the
column is out of sample.

It also does NOT use the shipped checkpoint. That one's dataset vocabulary is
{arxiv, govreport, humaneval, mbpp, sharegpt}, while the scenarios also draw
cnn, openorca and gsm8k -- it covers `long_*` fully, `mixed_*` partly and
`mlperf_*` not at all. Same architecture and same three features (log1p(ISL), a
model embedding, a dataset embedding), refitted on the workload the paper
actually replays.

## workload_harness/

The vLLM runner, the PBS job generation for the cluster, and the dataset prep.
It produced `results/workload/` and is not otherwise used.


## What is not here

`validate_latency.py` moved to `datacenter/perf_models/lookup/`. It scores TTFT
and TPOT predictions against the measured benchmarks and has nothing to do with
output length; it sat here only because it borrows one directory parser from
`benchmark_loader.py`, which it still imports.

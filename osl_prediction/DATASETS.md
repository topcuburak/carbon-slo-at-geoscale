# Datasets

## What the paper actually replays

Read off `loadgen/scenarios/*.csv`, the traces the simulator serves:

| family | datasets |
|---|---|
| `mixed_*`  | cnn, govreport, gsm8k, humaneval, mbpp, openorca |
| `mlperf_*` | cnn |
| `long_*`   | govreport |

Six in total. These are the `dataset` column on every request, and they are what
`fit_scenarios.py`'s dataset embedding is fitted over — which is why the shipped
predictor checkpoint cannot be used for the scenarios: its vocabulary is
{arxiv, govreport, humaneval, mbpp, sharegpt} and has never seen cnn, openorca
or gsm8k.

## What the workload harness could prepare

`workload_harness/dataset_prep/` had one script per dataset, and
`llm_calls.py` recognised these names:

    bbh
    codealpaca
    commonsense_qa      (also spelled commonsenseqa)
    govreport
    gsm8k
    narrativeqa
    squad_v2

Two of the seven -- govreport and gsm8k -- are in the traces. The other five
were never used by anything in this repository.

## What was actually measured

`results/workload/`, one directory per (model, dataset):

    llama3_70B_govreport      llama3_8B_govreport       mixtral_8x7b_govreport
    llama3_70B_narrativeqa    llama3_8B_narrativeqa     ...

and, through `datacenter/perf_models/lookup/measured/summaries/`:

    Llama3.1-8B_{Arxiv, GovReport, HumanEval, MBPP, ShareGPT}
    Qwen2.5-7B_{GovReport, MBPP, ShareGPT}
    MoE_GovReport, MoE_8x7B_Arxiv, MoE_8x22_MBPP

Note these are the OLD dataset vocabulary -- arxiv, narrativeqa and sharegpt
appear here and in no trace. The measurement era and the replay era used
different workloads, which is the reason the shipped checkpoint had to be
refitted.

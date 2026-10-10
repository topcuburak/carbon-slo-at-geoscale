# Dissecting the Carbon–SLO Tension in Geo-Distributed LLM Serving

Code and data for the paper *Dissecting the Carbon–SLO Tension in Geo-Distributed LLM
Serving*. It contains the geo-distributed LLM serving simulator, the
SLO- and carbon-aware schedulers, the configurations of every experiment in the paper, and
the scripts and aggregated data that produce its figures.

Paper: arXiv link coming soon.

## Architecture

```
                     ┌─────────────┐
                     │ Provisioner │  polls every DC each Δp and
                     └──────┬──────┘  pushes a snapshot to the schedulers
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌───────────┐       ┌───────────┐       ┌───────────┐
  │ scheduler │       │ scheduler │  ...  │ scheduler │   one per region;
  │ eu_north  │       │ us_east   │       │ asia      │   may forward to a
  └─────┬─────┘       └─────┬─────┘       └─────┬─────┘   remote region (+RTT)
        ▼                   ▼                   ▼
   DCs in region       DCs in region       DCs in region
   (nodes, engines,    continuous batching, KV cache,
    per-step latency and energy, PUE, grid CI)
```

The processes talk over gRPC:

| Process | Instances | Role |
|---|---|---|
| Scheduler | 1 per region | Receives requests from the load generator and routes each to a DC with the configured policy |
| Datacenter | 1 per DC | Simulates its nodes and inference engines step by step and accounts energy and carbon |
| Load generator | 1 or more shards | Replays the request trace as Poisson arrivals and records per-request TTFT, TPOT, and placement |

## Default topology

The main experiments (`scripts/sweep/topologies/v8_grid.json`) use 4 regional schedulers and
16 DCs with 4 nodes each:

| Region | DCs |
|---|---|
| `europe_north` | Netherlands, Belgium, Lithuania, Latvia |
| `us_east` | US_MISO, US_DukeProgressE, US_PJM, US_DukeCarolinas |
| `asia` | JP_Chubu, AU_VIC, JP_Chugoku, JP_Tokyo |
| `south_america` | BR_South, BR_North, BR_Central, Peru |

Each DC maps to an Electricity Maps zone; the simulator reads the zone's hourly direct carbon
intensity (gCO₂eq/kWh) at the simulated hour (`sim_datetime`).

Inter-region round-trip times (ms), one-year p50 between AWS regions from
[CloudPing](https://www.cloudping.co/grid); a forwarded request pays one RTT end to end:

|  | europe_north | us_east | asia | south_america |
|--|--:|--:|--:|--:|
| **europe_north** | 2 | 85 | 175 | 200 |
| **us_east** | 85 | 2 | 190 | 120 |
| **asia** | 175 | 190 | 2 | 280 |
| **south_america** | 200 | 120 | 280 | 10 |

The full matrix for all eight regions is in `common/network.py`.

## Repository layout

| Path | Contents |
|---|---|
| `datacenter/` | DC, node, and inference-engine processes: continuous batching, KV-cache admission, queueing, per-step latency and energy accounting |
| `datacenter/perf_models/analytical/` | Calibrated latency and power models (Section III-B); `fit_data/*.json` holds the fitted coefficients for every (model, TP) setup |
| `datacenter/perf_models/lookup/` | Legacy lookup backend and the calibration campaign drivers |
| `scheduler/` | Regional schedulers and all policies (`scheduler/policies/`): greedy (`carbon_aware`), standard ILP (`ilp_carbon_sla`), iterative ILP (`iterative_ilp_carbon_sla`), facility MILP (`facility_ilp_carbon_sla`), Green-LLM port (`green_llm.py`), RR, and LL |
| `provisioner/` | Global provisioner that polls DC state every Δp and broadcasts the snapshot |
| `loadgen/` | Poisson load generator; `loadgen/scenarios/` holds the request traces (token lengths and predicted OSLs only, no prompt text) |
| `common/` | Shared code: carbon-intensity loader (`carbon.py`), inter-region RTT matrix, PUE tables (`pue_by_dc.json`, `pue_by_dc_real.json`) |
| `carbon_intensity/` | Hourly grid carbon intensity (2021–2024) for the zones used in the paper, from Electricity Maps |
| `osl_prediction/` | Lightweight MLP output-length predictor and the script that writes `predicted_osl` into the traces |
| `proto/` | gRPC interface (`inference.proto`); Python stubs are generated automatically on the first run |
| `configs/` | The configuration of each experiment family behind Figures 4–7 |
| `scripts/sweep/` | Sweep driver (`sweep_all_policies.sh`) and the topologies (`topologies/*.json`) |
| `scripts/reproduce/` | `run_config.py`: runs one experiment family from a file in `configs/` |
| `scripts/test/` | Unit and integration tests (`run_all.sh`) |
| `figures/sweep/` | `summarize_cells.py` and the plotting scripts for Figures 4–7 |
| `data/figures/` | Aggregated per-cell results from our runs, read by the plotting scripts |

## Setup

Python 3.10 or newer on Linux.

```bash
pip install -r requirements.txt
```

The simulator runs in real time: each DC, scheduler, provisioner, and load-generator shard
is a separate process, and engines sleep for each predicted step. A 200 s policy–rate cell
therefore takes a little over 200 s. We ran all experiments on a single 64-core server, one
sweep at a time; running several sweeps concurrently on one host inflates latencies.

## Quick start

A 15 s run of the facility MILP for Llama-3.1-8B at 200 req/s:

```bash
DURATION=15 python scripts/reproduce/run_config.py configs/fig4_llama_ilp_rr.json \
    --results results/smoke --policies "facility_ilp_carbon_sla" --rates "200"
python figures/sweep/summarize_cells.py --model llama3_8B --tp 1 --root results/smoke \
    --tpot 100 --out results/smoke/cells.csv
```

## Reproducing the paper

### 1. Figures from our aggregated results

`data/figures/` contains the per-cell summaries of our runs, so the figures can be
regenerated directly:

```bash
cd figures/sweep
python plot_ipdps_frontier.py --out <dir>          # Figure 4 (carbon–SLO and TTFT tail)
python plot_ipdps_sensitivity.py --out fig5.png    # Figure 5 (PUE and CI sensitivity)
python plot_ipdps_system.py --out fig6.png         # Figure 6 (overhead, polling, topology)
python plot_ipdps_workload.py --out fig7.png       # Figure 7 (OSL prediction, prompt length)
```

The simulator validation numbers in Section V-B are in `data/figures/within_node_tp1.csv`.

### 2. Rerunning the experiments

Each file in `configs/` records the exact settings of one experiment family, including its
topology and trace. Run it with:

```bash
python scripts/reproduce/run_config.py configs/<config>.json --results results/<name>
```

`--policies` and `--rates` restrict a run to a subset, and `--dry-run` prints the
environment without launching. Then aggregate each results directory into a CSV with
`figures/sweep/summarize_cells.py` (use `--tpot 200` for Mixtral-8x7B) and point the
plotting scripts at it.

| Figure | Configs |
|---|---|
| Fig. 4 (main comparison) | `fig4_{llama,qwen,mixtral}_ilp_rr.json` (RR, LL, ILPs, MILP), `fig4_*_greedy.json` (greedy without batch capacity), `fig4_*_greenllm.json` (Green-LLM) |
| Fig. 5a (PUE) | `fig5a_{llama,mixtral}_pue_{uniform,real}.json` |
| Fig. 5b (CI hours) | `fig5b_*_ci_hour_example.json`; set `sim_datetime` to each hour listed in `data/figures/ci_hour_cv.csv` |
| Fig. 6a (overhead) | Recorded in the Fig. 4 runs (`sched_p50/p90/p99` columns) |
| Fig. 6b (polling) | `fig6b_*_poll_{sync,async}_example.json`; set `poll_period_s` to 0.5, 1, 2, 5, and 10 |
| Fig. 6c (topology) | `fig6c_llama_topology_{8sched,8dc}.json`; the base topology is `fig5a_llama_pue_uniform.json` |
| Fig. 7a (OSL) | `fig7a_{llama,mixtral}_predicted_osl.json`; the oracle side is the Fig. 4 runs |
| Fig. 7b (prompt length) | `fig7b_mixtral_{long,mixed}_example.json`; set `bs_cap` to 12–32 (long) and 16–64 (mixed) |

The main comparison (Fig. 4) and the polling, OSL, and prompt-length studies use the base
topology `scripts/sweep/topologies/v8_grid.json`. The PUE, CI, and topology studies use a
variant (`v9_*.json`) in which the three DCs without public PUE data are replaced by the
nearest-CI sites in the same regions.

### Calibration

The latency and power models are fitted to vLLM v0.6.3 measurements on A100-40GB GPUs; the
fitted coefficients are in `datacenter/perf_models/analytical/fit_data/` and are all the
simulator needs. The raw measurement logs are large and contain host-specific paths, so they
are not part of this repository; we plan to release them later.

## Data sources

- **Carbon intensity:** hourly zone-level data from [Electricity Maps](https://www.electricitymaps.com),
  included for the zones used in the paper and subject to Electricity Maps' terms of use.
- **Request traces:** token lengths derived from CNN-DailyMail, OpenOrca, GovReport, GSM8K,
  HumanEval, and MBPP; the traces contain no prompt or response text.
- **Inter-region RTT:** one-year p50 latencies between AWS regions from CloudPing.

## Citation

A BibTeX entry will be added with the arXiv link.

## License

See [LICENSE](LICENSE). The carbon-intensity data in `carbon_intensity/` is subject to
Electricity Maps' terms of use.

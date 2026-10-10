# Sweeps, analysis, and the things that are not launchers

Each component starts itself now:

    datacenter/start.sh    the fleet, one process per site
    scheduler/start.sh     one per group, from scheduler/fleet.yaml
    provisioner/start.sh   the poll-and-push loop
    loadgen/start.sh       the requests

In that order: each writes into results/logs/endpoints.json and the
next one reads it.

What is left here drives them -- the sweeps -- or reads what they
produced -- the analysis and plotting. `test/` checks them and
`benchmarks/` runs the simulator in process.

---

# Running the Policy Sweep Tests

## Quick Start

```bash
# Default: 50s duration, 25 req/s, 10 test cases
bash scripts/test/sweep_test.sh

# Custom duration and arrival rate
bash scripts/test/sweep_test.sh 120 40    # 120s at 40 req/s
```

Results go to `/tmp/grpc_sweep_results/` (CSVs + energy JSONs).
Logs go to `/tmp/grpc_test_logs/`.

## Sweep Script Parameters

These are set at the top of `scripts/test/sweep_test.sh`:

| Variable | Default | Description |
|---|---|---|
| `DURATION` | `50` | Test duration in seconds (1st CLI arg) |
| `ARRIVAL_RATE` | `25` | Poisson arrival rate in req/s (2nd CLI arg) |
| `CONCURRENT` | `5000` | Max in-flight requests in loadgen |
| `POLL_PERIOD_S` | `2` | Provisioner snapshot push interval (env var) |
| `LATENCY_CSV` | `datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv` | Path to latency benchmark data |
| `POWER_CSV` | `datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/power.csv` | Path to GPU power data |
| `ENDPOINTS` | `results/logs/endpoints.json` | Endpoint discovery file, written by the four launchers |
| `LOGDIR` | `/tmp/grpc_test_logs` | Per-service log directory |
| `RESULTS_DIR` | `/tmp/grpc_sweep_results` | Output directory for results |

### Changing Latency/Power Traces

To use different benchmark data (e.g., a different model or GPU):

```bash
# Edit sweep_test.sh lines 21-22:
LATENCY_CSV="path/to/your_benchmark.csv"
POWER_CSV="path/to/your_power.csv"
```

**Latency CSV format** (from `scripts/benchmark_vllm.py`):
```
sample_id,batch_size,in_tokens,out_tokens,ttft,tpot,inf_latency
0,1,512,128,0.045,0.0012,0.198
...
```

**Power CSV format**:
```
batch_size,power_w
1,627.0
2,635.0
...
```

## Datacenter Configuration (Environment Variables)

Each DC server reads these env vars. To change them globally,
edit the `start_dcs()` function in `sweep_test.sh`:

| Env Var | Default | Description |
|---|---|---|
| `DC_ID` | `dc-0` | Datacenter identifier (e.g., Paris, Tokyo) |
| `NUM_NODES` | `2` | Number of compute nodes per DC |
| `NUM_GPUS_PER_NODE` | `8` | Physical GPUs per node |
| `TP_SIZE` | `2` | Tensor parallel degree (GPUs per request) |
| `CARBON_INTENSITY` | `0.0` | Carbon intensity in gCO2eq/kWh |
| `LATENCY_CSV` | (required) | Benchmark CSV for TPOT/TTFT lookup |
| `POWER_CSV` | (optional) | Power CSV for energy tracking |
| `GRPC_PORT` | `50051` | Port to listen on |
| `GRPC_MAX_WORKERS` | `2000` | gRPC thread pool size |

### Changing TP Size

TP size determines how many physical GPUs serve one request.
`gpu_pairs = NUM_GPUS_PER_NODE / TP_SIZE`.

```bash
# Example: TP=4 with 8 GPUs → 2 GPU groups per node
# Edit start_dcs() in sweep_test.sh, add:
TP_SIZE=4 \
```

Current default: `TP_SIZE=2`, `NUM_GPUS_PER_NODE=8` → 4 GPU pairs per node.

**Important**: TP size changes the batch-size dynamics because fewer GPU
groups means more requests per group. Make sure your latency/power CSVs
were benchmarked with the same TP configuration.

### Changing Number of Nodes per DC

```bash
# In start_dcs(), change NUM_NODES:
NUM_NODES=2 \    # 2 nodes × 4 GPU pairs = 8 GPU pairs per DC
```

## Scheduler Configuration

| Env Var | Default | Description |
|---|---|---|
| `SCHED_ID` | (set by sweep) | Scheduler identifier |
| `SCHEDULER_GROUP` | `europe` | Region group: europe, us_east, us_west, asia |
| `ROUTING_POLICY` | (set by sweep) | Policy: round_robin, round_robin_local, least_loaded, least_loaded_local, weighted_least_loaded, carbon_aware |
| `ROUTING_ALPHA` | `0.5` | Carbon-aware tradeoff: 1.0=latency-only, 0.0=carbon-only |
| `SIMULATE_RTT` | `true` | Add simulated network delay for cross-region forwarding |
| `GRPC_PORT` | `50052` | Port to listen on |

## Provisioner Configuration

| Env Var | Default | Description |
|---|---|---|
| `POLL_PERIOD_S` | `2` | Seconds between polling DCs and pushing snapshots |
| `GRPC_PORT` | `50300` | Port to listen on |
| `LATENCY_CSV` | (required) | Same benchmark CSV (used for queue delay estimation) |

## Loadgen (Load Generator) Configuration

| CLI Arg | Env Var | Default | Description |
|---|---|---|---|
| `--duration` | `DURATION_S` | `30` | Test duration in seconds |
| `--arrival-distribution` | `ARRIVAL_DISTRIBUTION` | `uniform` | `uniform` or `poisson` |
| `--arrival-rate` | `ARRIVAL_RATE` | `13.3` | Requests/sec (Poisson mode) |
| `--min-interval-ms` | `MIN_INTERVAL_MS` | `30` | Min inter-arrival ms (uniform mode) |
| `--max-interval-ms` | `MAX_INTERVAL_MS` | `120` | Max inter-arrival ms (uniform mode) |
| `--trace-csv` | `TRACE_CSV` | (none) | CSV trace file for replay workload |
| `--min-in-tok` | `MIN_IN_TOK` | `200` | Min input tokens (random mode) |
| `--max-in-tok` | `MAX_IN_TOK` | `1200` | Max input tokens (random mode) |
| `--min-out-tok` | `MIN_OUT_TOK` | `50` | Min output tokens (random mode) |
| `--max-out-tok` | `MAX_OUT_TOK` | `300` | Max output tokens (random mode) |
| `--concurrent` | `CONCURRENT` | `16` | Max in-flight requests |
| `--seed` | `SEED` | `42` | Random seed |
| `--output` | `OUTPUT_CSV` | `/tmp/loadgen_results.csv` | Output CSV path |

## Energy and Carbon Model

Energy is tracked by a background sampler thread in each DC (1s interval):
- **Active GPU**: power = `power_w(batch_size)` from power CSV (interpolated)
- **Idle GPU**: power = 60W constant
- **Accumulation**: `energy_j += power × dt` every second

Carbon conversion:
```
carbon_g = energy_j × PUE × carbon_intensity / 3,600,000
```

| Env Var | Default | Description |
|---|---|---|
| `PUE` | `1.1` | Power Usage Effectiveness (cooling overhead) |
| `CARBON_INTENSITY` | `0.0` | gCO2eq/kWh for this DC |

## Carbon-Aware Scoring

The carbon-aware policy scores each DC as:
```
score = α × normalized_latency + (1 - α) × normalized_carbon
```

Additional parameters:

| Env Var | Default | Description |
|---|---|---|
| `ROUTING_ALPHA` | `0.5` | Tradeoff weight (1.0 = pure latency, 0.0 = pure carbon) |
| `REF_ISL` | `7500` | Reference input sequence length for normalization |
| `REF_OSL` | `500` | Reference output sequence length for normalization |

## Modifying Test Cases

Edit the `TESTS` array in `sweep_test.sh` (line 117):

```bash
# Format: "tag|policy|alpha"
TESTS=(
    "round_robin|round_robin|0"
    "carbon_aware_a0.5|carbon_aware|0.5"
    # Add your own...
)
```

Available policies: `round_robin`, `round_robin_local`, `least_loaded`,
`least_loaded_local`, `weighted_least_loaded`, `carbon_aware`.

## Modifying DC Topology

Carbon intensities are in the `CI` array (line 32). Port assignments
are in `DC_PORTS` (line 38) and `SCHED_PORTS` (line 44). The DC list
is in `start_dcs()` (line 55). To add/remove DCs:

1. Add/remove entries in `CI[]` and `DC_PORTS[]`
2. Add/remove DC names from the loop in `start_dcs()`
3. Restart the fleet so `results/logs/endpoints.json` is rewritten to match
4. Update `common/network.py` if the DC maps to a new region group

## Output Files

Per test case `TAG`:
- `/tmp/grpc_sweep_results/TAG.csv` — per-request latency results
- `/tmp/grpc_sweep_results/TAG_energy.json` — per-DC energy and carbon breakdown

### Results CSV columns
```
req_id, scheduler, dc_id, gpu_id, status, wall_ms, service_ms, in_tok, out_tok
```

### Energy JSON structure
```json
{
  "pue": 1.1,
  "per_dc": {
    "Paris": {"energy_j": 271629, "carbon_intensity": 6.8, "pue": 1.1, "carbon_g": 0.56},
    ...
  },
  "total_energy_j": 5050657,
  "total_energy_kwh": 1.4030,
  "total_carbon_g": 317.8
}
```

## Example: Full Custom Run

```bash
# TP=4, 2 nodes/DC, 120s test at 40 req/s, different traces
# Step 1: Edit sweep_test.sh
LATENCY_CSV="path/to/new_benchmark.csv"
POWER_CSV="path/to/new_power.csv"

# Step 2: In start_dcs(), set:
NUM_NODES=2 \
TP_SIZE=4 \

# Step 3: Run
bash scripts/test/sweep_test.sh 120 40
```

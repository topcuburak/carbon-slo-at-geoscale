"""Load real inference traces from CSV log files.

CSV files produced by real inference runs have the schema:

    sample_id, prompt, ref, gen, in_tokens, out_tokens, inf_latency

The filename encodes model, dataset, and batch size, e.g.
``meta-llama_Meta-Llama-3.1-8B-Instruct__govreport__BS_2.csv``

This module parses those files into ``InferenceRecord`` objects that the
load generator can replay.
"""

import csv
from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class InferenceRecord:
  """A single inference request extracted from a real run."""
  sample_id: int
  in_tokens: int
  out_tokens: int
  dataset: str = ""
  # What a length predictor said this request would generate, if the trace
  # carries the column. 0 when it does not. out_tokens stays ground truth.
  predicted_osl: int = 0


def load_trace(csv_path: str) -> List[InferenceRecord]:
  """Read a single CSV file and return a list of InferenceRecords."""
  records: List[InferenceRecord] = []

  with open(csv_path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      in_tok = int(row.get("in_tokens") or row.get("in_tok") or 0)
      out_tok = int(row.get("out_tokens") or row.get("out_tok") or 0)
      dataset = row.get("dataset", "")
      predicted = int(row.get("predicted_osl") or 0)
      records.append(InferenceRecord(
        sample_id=int(row["sample_id"]),
        in_tokens=in_tok,
        out_tokens=out_tok,
        dataset=dataset,
        predicted_osl=predicted,
      ))
  return records


def make_trace_generator(
  records: List[InferenceRecord],
) -> Callable[[], InferenceRecord]:
  """Return a callable handing out one record per call, in file order.

  When the run outlasts the trace it wraps to the beginning, so a request is
  dispatched more than once -- at the paper's rates the trace is traversed 1.7
  to 2.9 times. The cycle is deterministic, which is what makes a run
  reproducible and what lets a dispatch be mapped back to its sample:

      sample_id = (req_num - 1) mod len(records) + 1

  It also means the id on the wire cannot be the trace's own: several identical
  requests are in flight at once, and only the client's dispatch counter tells
  them apart.
  """
  if not records:
    raise ValueError("Cannot create trace generator from empty record list")
  _idx = [0]

  def _gen() -> InferenceRecord:
    rec = records[_idx[0] % len(records)]
    _idx[0] += 1
    return rec
  return _gen

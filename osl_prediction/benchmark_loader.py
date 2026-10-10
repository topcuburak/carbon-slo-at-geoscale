"""Loads the benchmark CSVs that train the L1 output-length predictor.

Named for where the CSVs come from -- the latency benchmark runs -- which reads
as though this were a latency tool. It is not: `train_l1.py` takes its training
rows from here, and every row is a (prompt length, chosen output length) pair.

Loads merged_benchmark.csv files from datacenter/perf_models/lookup/measured/summaries/*, deduplicates
across batch sizes, and produces a DataFrame with:
    sample_id, in_tokens, out_tokens, model, dataset

Used for the L1 feature-based OSL predictor that predicts out_tokens
from (in_tokens, model, dataset) without needing prompt text.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# Maps directory names to (model_key, dataset)
_DIR_PATTERNS = [
    # Llama3.1-8B_GovReport, Llama3.1-8B_MBPP, etc.
    (re.compile(r"Llama3\.1-8B_(.+)"), "llama3_8B"),
    # MoE_GovReport (Mixtral 8x7B)
    (re.compile(r"MoE_GovReport"), "mixtral_8x7b"),
    (re.compile(r"MoE_8x7B_(.+)"), "mixtral_8x7b"),
    # MoE_8x22_MBPP (Mixtral 8x22B)
    (re.compile(r"MoE_8x22_(.+)"), "mixtral_8x22b"),
    # Qwen2.5-7B_GovReport, etc.
    (re.compile(r"Qwen2\.5-7B_(.+)"), "qwen2.5_7b"),
]


def _parse_benchmark_dir(name: str):
    """Extract (model_key, dataset) from a benchmark directory name."""
    for pattern, model_key in _DIR_PATTERNS:
        m = pattern.match(name)
        if m:
            # Dataset from capture group or from dir name
            if m.lastindex and m.lastindex >= 1:
                dataset = m.group(1).lower()
            else:
                # e.g. MoE_GovReport -> govreport
                dataset = name.split("_", 1)[1].lower() if "_" in name else name.lower()
            return model_key, dataset
    return None, None


def load_benchmark_data(
    inferences_dir: str,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load benchmark CSVs and deduplicate across batch sizes.

    For each unique sample_id within a (model, dataset), takes the
    median out_tokens across batch sizes (OSL is ~91% stable across BS).

    Args:
        inferences_dir: Path to datacenter/perf_models/lookup/measured/summaries/
        models: If given, only load these model keys.
        datasets: If given, only load these dataset names.

    Returns:
        DataFrame with: sample_id, in_tokens, out_tokens, model, dataset
    """
    inferences_dir = Path(inferences_dir)
    all_rows = []

    for subdir in sorted(inferences_dir.iterdir()):
        if not subdir.is_dir():
            continue

        model_key, dataset = _parse_benchmark_dir(subdir.name)
        if model_key is None:
            continue
        if models and model_key not in models:
            continue
        if datasets and dataset not in datasets:
            continue

        csv_path = subdir / "merged_benchmark.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if "sample_id" not in df.columns or "out_tokens" not in df.columns:
            continue

        # Deduplicate: median out_tokens per sample_id
        grouped = df.groupby("sample_id").agg(
            in_tokens=("in_tokens", "first"),
            out_tokens=("out_tokens", "median"),
        ).reset_index()

        grouped["out_tokens"] = grouped["out_tokens"].astype(int)
        grouped["model"] = model_key
        grouped["dataset"] = dataset
        all_rows.append(grouped)

    if not all_rows:
        return pd.DataFrame(columns=["sample_id", "in_tokens", "out_tokens", "model", "dataset"])

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["model", "dataset", "sample_id"]).reset_index(drop=True)
    return result


def get_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    split_by: str = "sample",
):
    """Split data into train/val/test.

    Args:
        split_by: "sample" = stratified random, "dataset" = hold out datasets.
    """
    rng = np.random.RandomState(seed)

    if split_by == "dataset":
        ds_list = df["dataset"].unique().tolist()
        rng.shuffle(ds_list)
        if len(ds_list) < 3:
            raise ValueError(f"Need >=3 datasets for dataset-split, got {ds_list}")
        test_ds = ds_list[0]
        val_ds = ds_list[1]
        train_ds = ds_list[2:]
        test_df = df[df["dataset"] == test_ds]
        val_df = df[df["dataset"] == val_ds]
        train_df = df[df["dataset"].isin(train_ds)]
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_test = int(len(df) * test_frac)
        n_val = int(len(df) * val_frac)
        test_df = df.iloc[indices[:n_test]]
        val_df = df.iloc[indices[n_test:n_test + n_val]]
        train_df = df.iloc[indices[n_test + n_val:]]

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect benchmark data")
    parser.add_argument("--inferences-dir", type=str,
                        default="datacenter/perf_models/lookup/measured/summaries",
                        help="Path to datacenter/perf_models/lookup/measured/summaries/")
    args = parser.parse_args()

    df = load_benchmark_data(args.inferences_dir)
    print(f"Loaded {len(df)} unique samples\n")
    print(f"Models:   {df['model'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}\n")
    for (m, d), g in df.groupby(["model", "dataset"]):
        osl = g["out_tokens"]
        it = g["in_tokens"]
        print(f"  {m:20s} {d:15s}  n={len(g):5d}  "
              f"in_tok: mean={it.mean():.0f} std={it.std():.0f}  "
              f"out_tok: mean={osl.mean():.0f} std={osl.std():.0f} "
              f"p50={osl.median():.0f} p90={osl.quantile(0.9):.0f}")

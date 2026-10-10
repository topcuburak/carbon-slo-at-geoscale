"""Unified data loader for OSL prediction.

Loads inference result CSVs across models and datasets, deduplicates
samples across batch sizes (same prompt seen at multiple BS values),
and produces a clean DataFrame with columns:
    prompt, in_tokens, out_tokens, model, dataset

OSL (out_tokens) is taken as the median across batch sizes for each
unique (sample_id, model, dataset) triple — since ~91% of samples
have identical OSL across BS, median is robust to the rare divergences.
"""

import csv
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# Maps directory names to (model_name, dataset_name)
_DIR_PATTERN = re.compile(
    r"(?P<model>llama3_8B|llama3_70B|mixtral_8x7b|qwen2\.5_7b)_(?P<dataset>.+)"
)

# Friendly model names for features / display
MODEL_NAMES = {
    "llama3_8B": "LLaMA-3.1-8B",
    "llama3_70B": "LLaMA-3.1-70B",
    "mixtral_8x7b": "Mixtral-8x7B",
    "qwen2.5_7b": "Qwen-2.5-7B",
}


def _parse_dir_name(name: str):
    """Extract (model_key, dataset) from a results subdirectory name."""
    m = _DIR_PATTERN.match(name)
    if m:
        return m.group("model"), m.group("dataset")
    return None, None


def load_inference_results(
    results_dir: str,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    max_prompt_tokens: int = 0,
) -> pd.DataFrame:
    """Load and deduplicate inference results into a unified DataFrame.

    Args:
        results_dir: Path to the results/workload/ directory.
        models: If given, only load these model keys (e.g. ["llama3_8B"]).
        datasets: If given, only load these dataset names (e.g. ["govreport"]).
        max_prompt_tokens: If > 0, drop prompts longer than this (for memory).

    Returns:
        DataFrame with columns:
            sample_id, prompt, in_tokens, out_tokens, model, dataset
        Deduplicated: one row per unique (sample_id, model, dataset).
    """
    results_dir = Path(results_dir)
    all_rows = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        model_key, dataset = _parse_dir_name(subdir.name)
        if model_key is None:
            continue
        if models and model_key not in models:
            continue
        if datasets and dataset not in datasets:
            continue

        csv_files = sorted(subdir.glob("*.csv"))
        if not csv_files:
            continue

        # Collect all rows across batch-size CSVs
        sample_data = {}  # sample_id -> {prompt, in_tokens, out_tokens_list}
        for csv_file in csv_files:
            try:
                reader = csv.DictReader(open(csv_file, encoding="utf-8"))
                if "sample_id" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    sid = row["sample_id"]
                    ot = int(row["out_tokens"])
                    if sid not in sample_data:
                        sample_data[sid] = {
                            "prompt": row["prompt"],
                            "in_tokens": int(row["in_tokens"]),
                            "out_tokens_list": [],
                        }
                    sample_data[sid]["out_tokens_list"].append(ot)
            except Exception:
                continue

        # Deduplicate: median OSL across batch sizes
        for sid, data in sample_data.items():
            in_tok = data["in_tokens"]
            if max_prompt_tokens > 0 and in_tok > max_prompt_tokens:
                continue
            median_osl = int(np.median(data["out_tokens_list"]))
            all_rows.append({
                "sample_id": sid,
                "prompt": data["prompt"],
                "in_tokens": in_tok,
                "out_tokens": median_osl,
                "model": model_key,
                "dataset": dataset,
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Sort for reproducibility
    df = df.sort_values(["model", "dataset", "sample_id"]).reset_index(drop=True)
    return df


def create_osl_buckets(
    out_tokens: pd.Series,
    n_buckets: int = 8,
    strategy: str = "quantile",
) -> tuple:
    """Create OSL bucket labels from continuous out_tokens.

    Args:
        out_tokens: Series of output token counts.
        n_buckets: Number of buckets.
        strategy: "quantile" for equal-frequency, "uniform" for equal-width.

    Returns:
        (labels, bin_edges) where labels is an integer Series [0, n_buckets).
    """
    if strategy == "quantile":
        bin_edges = np.quantile(
            out_tokens, np.linspace(0, 1, n_buckets + 1)
        )
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(
            out_tokens.min(), out_tokens.max() + 1, n_buckets + 1
        )

    labels = np.digitize(out_tokens, bin_edges[1:-1])  # [0, len(edges)-2)
    return pd.Series(labels, index=out_tokens.index), bin_edges


def get_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 42,
    split_by: str = "sample",
):
    """Split data into train/val/test.

    Args:
        df: Full DataFrame.
        test_frac: Fraction for test.
        val_frac: Fraction for validation.
        seed: Random seed.
        split_by: "sample" = random split, "dataset" = hold out entire
                  datasets for cross-dataset generalization evaluation.

    Returns:
        (train_df, val_df, test_df)
    """
    rng = np.random.RandomState(seed)

    if split_by == "dataset":
        # Hold out one dataset as test, one as val
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
        # Stratified random split by (model, dataset)
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

    parser = argparse.ArgumentParser(description="Load and inspect OSL data")
    parser.add_argument("--results-dir", type=str,
                        default="results/workload",
                        help="Path to results/workload/")
    args = parser.parse_args()

    df = load_inference_results(args.results_dir)
    print(f"Loaded {len(df)} unique samples")
    print(f"\nModels:   {df['model'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"\nPer (model, dataset):")
    for (m, d), g in df.groupby(["model", "dataset"]):
        osl = g["out_tokens"]
        print(f"  {m:20s} {d:15s}  n={len(g):5d}  "
              f"mean={osl.mean():.0f}  std={osl.std():.0f}  "
              f"p50={osl.median():.0f}  p90={osl.quantile(0.9):.0f}")

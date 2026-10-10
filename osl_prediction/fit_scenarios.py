"""Fill predicted_osl on the scenario traces, out of fold.

The shipped checkpoint cannot be used here. Its dataset vocabulary is
{arxiv, govreport, humaneval, mbpp, sharegpt} and the scenarios also draw on
cnn, openorca and gsm8k, which it has never seen -- so it covers long_* fully,
mixed_* partly and mlperf_* not at all. It is the same architecture and the
same three features, retrained on the workload the paper actually replays.

CROSS-FITTED, and that is the whole point. A predictor trained on all the rows
and then asked about those rows is reporting memorisation: predicted_osl would
approach out_tok and the OSL sensitivity arm would conclude that a predictor is
as good as an oracle, which is a result about the fit and not about scheduling.
Here the rows are split into K folds and each fold's predictions come from a
model that trained on the other K-1. Every value in the column is out of
sample.

Features are the L1 model's: log1p(ISL), a model embedding and a dataset
embedding. No prompt text -- the shipped checkpoint does not use any either,
though predict.py, which targets the unbuilt L2/L3 transformer, does.

    python -m osl_prediction.fit_scenarios
"""

import argparse
import csv
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from osl_prediction.train_l1 import (L1Predictor, BenchmarkDataset, PinballLoss,
                                    train_epoch)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MODELS = ("llama3_8B", "qwen2.5_7B", "mixtral_8x7b")


def load_scenarios(scen_dir):
    """Every scenario CSV as one frame, remembering where each row came from."""
    rows = []
    for path in sorted(glob.glob(os.path.join(scen_dir, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if not name.startswith(("mlperf_", "mixed_", "long_")):
            continue
        family, model = name.split("_", 1)
        # mixed_trace*.csv share the "mixed_" prefix but are the earlier
        # generation, with no model in the name and measured rather than
        # tokenized OSL. They are not scenarios.
        if model not in MODELS:
            continue
        with open(path) as f:
            for i, r in enumerate(csv.DictReader(f)):
                rows.append({"path": path, "row": i, "model": model,
                             "dataset": r["dataset"],
                             "in_tokens": int(r["in_tok"]),
                             "out_tokens": int(r["out_tok"])})
    return pd.DataFrame(rows)


@torch.no_grad()
def predict(model, ds, device, batch_size=4096):
    model.eval()
    q_out, b_out = [], []
    for batch in DataLoader(ds, batch_size=batch_size):
        out = model(batch["in_tokens_log"].to(device),
                    batch["model_id"].to(device),
                    batch["dataset_id"].to(device))
        # The quantile head is fitted on log1p(OSL); the median is the point
        # estimate a scheduler would act on.
        q = out["quantiles"][:, QUANTILES.index(0.5)]
        q_out.append(torch.expm1(q).clamp(min=1).cpu().numpy())
        b_out.append(out["cls_logits"].argmax(dim=-1).cpu().numpy())
    return np.concatenate(q_out), np.concatenate(b_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scen-dir", default="loadgen/scenarios")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-buckets", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_scenarios(a.scen_dir)
    log.info("loaded %d rows from %d traces", len(df), df.path.nunique())
    for (m, d), g in df.groupby(["model", "dataset"]):
        log.info("  %-14s %-10s n=%-6d OSL p50 %5.0f", m, d, len(g),
                 g.out_tokens.median())

    edges = np.unique(np.quantile(df.out_tokens, np.linspace(0, 1, a.n_buckets + 1)))
    n_classes = len(edges) - 1
    model_to_id = {m: i for i, m in enumerate(sorted(df.model.unique()))}
    dataset_to_id = {d: i for i, d in enumerate(sorted(df.dataset.unique()))}
    log.info("buckets %s", [round(x) for x in edges])
    log.info("models %s | datasets %s", model_to_id, dataset_to_id)

    rng = np.random.RandomState(a.seed)
    fold = rng.randint(0, a.folds, size=len(df))
    pred = np.zeros(len(df)); bucket = np.zeros(len(df), dtype=int)

    for k in range(a.folds):
        tr, te = df[fold != k], df[fold == k]
        net = L1Predictor(len(model_to_id), len(dataset_to_id), mode="both",
                          n_classes=n_classes, quantiles=QUANTILES).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        loader = DataLoader(BenchmarkDataset(tr, model_to_id, dataset_to_id, edges),
                            batch_size=a.batch_size, shuffle=True)
        for ep in range(a.epochs):
            loss = train_epoch(net, loader, opt,
                               nn.CrossEntropyLoss().to(device),
                               PinballLoss(QUANTILES).to(device), "both", device)
        p, b = predict(net, BenchmarkDataset(te, model_to_id, dataset_to_id, edges), device)
        pred[fold == k] = p; bucket[fold == k] = b
        err = np.abs(p - te.out_tokens.values)
        log.info("fold %d: train %d, held out %d, last loss %.4f, "
                 "held-out MAE %.0f tokens (median |err| %.0f)",
                 k, len(tr), len(te), loss, err.mean(), np.median(err))

    df["predicted_osl"] = np.rint(pred).astype(int)
    df["predicted_bucket"] = bucket
    e = np.abs(df.predicted_osl - df.out_tokens)
    log.info("OUT OF FOLD over all %d rows: MAE %.0f, median |err| %.0f, "
             "within 2x on %.1f%%", len(df), e.mean(), np.median(e),
             100 * ((df.predicted_osl <= 2 * df.out_tokens) &
                    (df.predicted_osl >= 0.5 * df.out_tokens)).mean())

    for path, g in df.groupby("path"):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        g = g.sort_values("row")
        for r, p, b in zip(rows, g.predicted_osl, g.predicted_bucket):
            r["predicted_osl"] = int(p)
            r["predicted_bucket"] = int(b)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()

"""Training pipeline for L1 feature-based OSL predictor.

Predicts out_tokens from (in_tokens, model, dataset) using a lightweight
MLP. No prompt text needed — uses benchmark CSVs.

Supports three modes:
  - classification: predict OSL bucket (cross-entropy loss)
  - regression: predict quantiles of out_tokens (pinball loss)
  - both: joint classification + quantile regression

Features:
  - in_tokens (log-scaled)
  - model embedding (learned)
  - dataset embedding (learned)

Usage:
    # Classification
    python -m osl_prediction.train_l1 --inferences-dir datacenter/perf_models/lookup/measured/summaries \\
        --mode classification --n-buckets 8 --epochs 200 \\
        --output-dir osl_prediction/checkpoints/L1_cls

    # Regression
    python -m osl_prediction.train_l1 --inferences-dir datacenter/perf_models/lookup/measured/summaries \\
        --mode regression --epochs 200 \\
        --output-dir osl_prediction/checkpoints/L1_reg

    # Both
    python -m osl_prediction.train_l1 --inferences-dir datacenter/perf_models/lookup/measured/summaries \\
        --mode both --epochs 200 \\
        --output-dir osl_prediction/checkpoints/L1_both
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from osl_prediction.benchmark_loader import load_benchmark_data, get_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("osl_train_l1")


# ── Model ────────────────────────────────────────────────────────────────

class L1Predictor(nn.Module):
    """Lightweight MLP for OSL prediction from (in_tokens, model, dataset).

    Supports classification (bucket), regression (quantile), or both.
    """

    def __init__(self, n_models: int, n_datasets: int,
                 mode: str = "both",
                 n_classes: int = 8,
                 quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                 hidden_dim: int = 128):
        super().__init__()
        self.mode = mode
        self.quantiles = quantiles

        self.model_embed = nn.Embedding(n_models, 32)
        self.dataset_embed = nn.Embedding(n_datasets, 32)

        input_dim = 1 + 32 + 32  # log(in_tokens) + model_embed + dataset_embed

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        if mode in ("classification", "both"):
            self.cls_head = nn.Linear(hidden_dim, n_classes)
        else:
            self.cls_head = None

        if mode in ("regression", "both"):
            self.q_head = nn.Linear(hidden_dim, len(quantiles))
        else:
            self.q_head = None

    def forward(self, in_tokens_log, model_ids, dataset_ids):
        m_emb = self.model_embed(model_ids)
        d_emb = self.dataset_embed(dataset_ids)
        x = torch.cat([in_tokens_log, m_emb, d_emb], dim=-1)
        h = self.backbone(x)

        result = {}
        if self.cls_head is not None:
            result["cls_logits"] = self.cls_head(h)
        if self.q_head is not None:
            result["quantiles"] = self.q_head(h)
        return result


# ── Losses ───────────────────────────────────────────────────────────────

class PinballLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, pred, target):
        target = target.unsqueeze(1).expand_as(pred)
        errors = target - pred
        loss = torch.max(self.quantiles * errors, (self.quantiles - 1) * errors)
        return loss.mean()


# ── Dataset ──────────────────────────────────────────────────────────────

class BenchmarkDataset(Dataset):
    def __init__(self, df, model_to_id, dataset_to_id, bin_edges=None):
        self.in_tokens_log = np.log1p(df["in_tokens"].values).astype(np.float32)
        self.out_tokens_log = np.log1p(df["out_tokens"].values).astype(np.float32)
        self.out_tokens_raw = df["out_tokens"].values.astype(np.float32)
        self.model_ids = df["model"].map(model_to_id).values.astype(np.int64)
        self.dataset_ids = df["dataset"].map(dataset_to_id).values.astype(np.int64)

        if bin_edges is not None:
            self.bucket_labels = np.digitize(
                self.out_tokens_raw, bin_edges[1:-1]
            ).astype(np.int64)
        else:
            self.bucket_labels = np.zeros(len(self.in_tokens_log), dtype=np.int64)

    def __len__(self):
        return len(self.in_tokens_log)

    def __getitem__(self, idx):
        return {
            "in_tokens_log": torch.tensor([self.in_tokens_log[idx]], dtype=torch.float32),
            "model_id": torch.tensor(self.model_ids[idx], dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_ids[idx], dtype=torch.long),
            "bucket_label": torch.tensor(self.bucket_labels[idx], dtype=torch.long),
            "osl_log": torch.tensor(self.out_tokens_log[idx], dtype=torch.float32),
            "osl_raw": torch.tensor(self.out_tokens_raw[idx], dtype=torch.float32),
        }


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, loss_fn_cls, loss_fn_q, mode, device):
    model.eval()
    total_loss = 0.0
    n_total = 0
    n_correct = 0
    all_preds = []
    all_targets = []
    all_preds_log = []
    all_targets_raw = []

    for batch in dataloader:
        in_tok = batch["in_tokens_log"].to(device)
        m_id = batch["model_id"].to(device)
        d_id = batch["dataset_id"].to(device)
        bucket_labels = batch["bucket_label"].to(device)
        osl_log = batch["osl_log"].to(device)
        osl_raw = batch["osl_raw"]

        out = model(in_tok, m_id, d_id)
        bs = len(osl_log)

        loss = 0.0
        if "cls_logits" in out and loss_fn_cls is not None:
            loss = loss + loss_fn_cls(out["cls_logits"], bucket_labels)
            preds = out["cls_logits"].argmax(dim=1)
            n_correct += (preds == bucket_labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(bucket_labels.cpu().numpy())

        if "quantiles" in out and loss_fn_q is not None:
            loss = loss + loss_fn_q(out["quantiles"], osl_log)
            all_preds_log.append(out["quantiles"].cpu().numpy())

        all_targets_raw.append(osl_raw.numpy())
        total_loss += (loss.item() if isinstance(loss, torch.Tensor) else loss) * bs
        n_total += bs

    metrics = {"loss": total_loss / max(n_total, 1)}

    # Classification metrics
    if all_preds:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        metrics["accuracy"] = n_correct / max(n_total, 1)
        metrics["accuracy_within_1"] = float(np.mean(np.abs(all_preds - all_targets) <= 1))

    # Regression metrics
    if all_preds_log:
        preds_log = np.concatenate(all_preds_log, axis=0)
        targets_raw = np.concatenate(all_targets_raw, axis=0)
        q_idx = preds_log.shape[1] // 2  # p50
        p50_pred_raw = np.expm1(preds_log[:, q_idx])
        ae = np.abs(targets_raw - p50_pred_raw)
        metrics["mae"] = float(ae.mean())
        metrics["median_ae"] = float(np.median(ae))
        nonzero = targets_raw > 0
        metrics["mape"] = float(np.mean(ae[nonzero] / targets_raw[nonzero]) * 100) if nonzero.any() else 0
        metrics["within_25pct"] = float(np.mean(ae <= np.maximum(targets_raw * 0.25, 10)))
        metrics["within_50pct"] = float(np.mean(ae <= np.maximum(targets_raw * 0.50, 10)))

    return metrics


# ── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, loss_fn_cls, loss_fn_q, mode, device):
    model.train()
    total_loss = 0.0
    n_total = 0

    for batch in dataloader:
        in_tok = batch["in_tokens_log"].to(device)
        m_id = batch["model_id"].to(device)
        d_id = batch["dataset_id"].to(device)
        bucket_labels = batch["bucket_label"].to(device)
        osl_log = batch["osl_log"].to(device)

        optimizer.zero_grad()
        out = model(in_tok, m_id, d_id)

        loss = 0.0
        if "cls_logits" in out and loss_fn_cls is not None:
            loss = loss + loss_fn_cls(out["cls_logits"], bucket_labels)
        if "quantiles" in out and loss_fn_q is not None:
            loss = loss + loss_fn_q(out["quantiles"], osl_log)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(osl_log)
        n_total += len(osl_log)

    return total_loss / max(n_total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train L1 feature-based OSL predictor")
    parser.add_argument("--inferences-dir", type=str, required=True,
                        help="Path to datacenter/perf_models/lookup/measured/summaries/")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--split-by", choices=["sample", "dataset"], default="sample")
    # Mode
    parser.add_argument("--mode", choices=["classification", "regression", "both"],
                        default="both")
    parser.add_argument("--n-buckets", type=int, default=8,
                        help="Number of OSL buckets (for classification)")
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 0.9])
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=128)
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="osl_prediction/checkpoints/L1")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Mode: %s", args.mode)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──
    log.info("Loading benchmark data from %s ...", args.inferences_dir)
    df = load_benchmark_data(args.inferences_dir, models=args.models,
                             datasets=args.datasets)
    log.info("Loaded %d unique samples", len(df))

    if len(df) == 0:
        log.error("No data loaded!")
        return

    for (m, d), g in df.groupby(["model", "dataset"]):
        log.info("  %s / %s  n=%d  out_tok: mean=%.0f std=%.0f p50=%.0f",
                 m, d, len(g), g["out_tokens"].mean(), g["out_tokens"].std(),
                 g["out_tokens"].median())

    # ── Bucket edges (for classification) ──
    bin_edges = None
    n_classes = args.n_buckets
    if args.mode in ("classification", "both"):
        bin_edges = np.quantile(df["out_tokens"], np.linspace(0, 1, args.n_buckets + 1))
        bin_edges = np.unique(bin_edges)
        n_classes = len(bin_edges) - 1
        log.info("OSL bucket edges: %s  (%d classes)", bin_edges.tolist(), n_classes)

    # ── Mappings ──
    model_keys = sorted(df["model"].unique())
    model_to_id = {m: i for i, m in enumerate(model_keys)}
    dataset_keys = sorted(df["dataset"].unique())
    dataset_to_id = {d: i for i, d in enumerate(dataset_keys)}
    log.info("Models: %s", model_to_id)
    log.info("Datasets: %s", dataset_to_id)

    # ── Split ──
    train_df, val_df, test_df = get_split(df, seed=args.seed, split_by=args.split_by)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # ── Datasets ──
    train_ds = BenchmarkDataset(train_df, model_to_id, dataset_to_id, bin_edges)
    val_ds = BenchmarkDataset(val_df, model_to_id, dataset_to_id, bin_edges)
    test_ds = BenchmarkDataset(test_df, model_to_id, dataset_to_id, bin_edges)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4,
                             num_workers=2, pin_memory=True)

    # ── Model ──
    quantiles = tuple(args.quantiles)
    model = L1Predictor(
        n_models=len(model_keys),
        n_datasets=len(dataset_keys),
        mode=args.mode,
        n_classes=n_classes,
        quantiles=quantiles,
        hidden_dim=args.hidden_dim,
    ).to(device)
    log.info("L1 model params: %d", sum(p.numel() for p in model.parameters()))

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Loss ──
    loss_fn_cls = nn.CrossEntropyLoss() if args.mode in ("classification", "both") else None
    loss_fn_q = PinballLoss(quantiles).to(device) if args.mode in ("regression", "both") else None

    # ── Training loop ──
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer,
                                 loss_fn_cls, loss_fn_q, args.mode, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn_cls, loss_fn_q,
                               args.mode, device)
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            parts = [f"Epoch {epoch}/{args.epochs}",
                     f"train_loss={train_loss:.4f}",
                     f"val_loss={val_metrics['loss']:.4f}"]
            if "accuracy" in val_metrics:
                parts.append(f"acc={val_metrics['accuracy']:.3f}")
                parts.append(f"acc±1={val_metrics['accuracy_within_1']:.3f}")
            if "mae" in val_metrics:
                parts.append(f"MAE={val_metrics['mae']:.1f}")
                parts.append(f"MAPE={val_metrics['mape']:.1f}%")
            parts.append(f"[{elapsed:.2f}s]")
            log.info("  ".join(parts))

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_to_id": model_to_id,
                "dataset_to_id": dataset_to_id,
                "quantiles": list(quantiles),
                "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
            }, os.path.join(args.output_dir, "best_model.pt"))

    # ── Test evaluation ──
    log.info("=" * 60)
    log.info("Loading best model for test evaluation...")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, loss_fn_cls, loss_fn_q,
                            args.mode, device)
    parts = ["TEST"]
    parts.append(f"loss={test_metrics['loss']:.4f}")
    if "accuracy" in test_metrics:
        parts.append(f"acc={test_metrics['accuracy']:.3f}")
        parts.append(f"acc±1={test_metrics['accuracy_within_1']:.3f}")
    if "mae" in test_metrics:
        parts.append(f"MAE={test_metrics['mae']:.1f}")
        parts.append(f"MedAE={test_metrics['median_ae']:.1f}")
        parts.append(f"MAPE={test_metrics['mape']:.1f}%")
        parts.append(f"within25%={test_metrics['within_25pct']:.3f}")
        parts.append(f"within50%={test_metrics['within_50pct']:.3f}")
    log.info("  ".join(parts))

    # ── Per-model/dataset breakdown ──
    log.info("Per-model/dataset test breakdown:")
    for (m, d), group in test_df.groupby(["model", "dataset"]):
        if len(group) < 5:
            continue
        sub_ds = BenchmarkDataset(group.reset_index(drop=True), model_to_id,
                                  dataset_to_id, bin_edges)
        sub_loader = DataLoader(sub_ds, batch_size=args.batch_size * 4, num_workers=2)
        sub_metrics = evaluate(model, sub_loader, loss_fn_cls, loss_fn_q,
                               args.mode, device)
        parts = [f"  {m} / {d}  (n={len(group)})"]
        if "accuracy" in sub_metrics:
            parts.append(f"acc={sub_metrics['accuracy']:.3f}")
            parts.append(f"acc±1={sub_metrics['accuracy_within_1']:.3f}")
        if "mae" in sub_metrics:
            parts.append(f"MAE={sub_metrics['mae']:.1f}")
            parts.append(f"MAPE={sub_metrics['mape']:.1f}%")
            parts.append(f"within25%={sub_metrics['within_25pct']:.3f}")
        log.info("  ".join(parts))

    # ── Save results ──
    results = {
        "config": config,
        "model_to_id": model_to_id,
        "dataset_to_id": dataset_to_id,
        "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
        "history": history,
        "test_metrics": test_metrics,
        "data_stats": {
            "total_samples": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "models": model_keys,
            "datasets": dataset_keys,
        },
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s/results.json", args.output_dir)


if __name__ == "__main__":
    main()

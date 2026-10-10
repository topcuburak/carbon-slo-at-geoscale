"""Training pipeline for L2/L3 OSL predictors.

Supports:
  - Cross-model training: single predictor across LLaMA, Mixtral, Qwen
  - Cross-dataset evaluation: train on some datasets, test on held-out
  - Both classification (bucket) and quantile regression heads
  - Configurable for server execution (all paths via CLI args)

Usage:
    # L3 full DistilBERT, all models, random split
    python -m osl_prediction.train --level L3 --results-dir results/workload \\
        --epochs 10 --batch-size 32 --output-dir osl_prediction/checkpoints

    # L2 lightweight, cross-dataset split
    python -m osl_prediction.train --level L2 --results-dir results/workload \\
        --split-by dataset --epochs 10 --output-dir osl_prediction/checkpoints

    # Single model only
    python -m osl_prediction.train --level L3 --models llama3_8B \\
        --results-dir results/workload --output-dir osl_prediction/checkpoints
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from osl_prediction.data_loader import (
    load_inference_results,
    create_osl_buckets,
    get_split,
)
from osl_prediction.models import build_l2_predictor, build_l3_predictor
from osl_prediction.losses import CombinedLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("osl_train")


# ── Dataset ──────────────────────────────────────────────────────────────

class OSLDataset(Dataset):
    """Tokenized prompt dataset for OSL prediction."""

    def __init__(self, df, tokenizer, max_length, model_to_id, dataset_to_id, bin_edges):
        self.prompts = df["prompt"].tolist()
        self.out_tokens = df["out_tokens"].values.astype(np.float32)
        self.model_ids = df["model"].map(model_to_id).values.astype(np.int64)
        self.dataset_ids = df["dataset"].map(dataset_to_id).values.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Bucket labels
        self.bucket_labels = np.digitize(
            self.out_tokens, bin_edges[1:-1]
        ).astype(np.int64)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "model_id": torch.tensor(self.model_ids[idx], dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_ids[idx], dtype=torch.long),
            "bucket_label": torch.tensor(self.bucket_labels[idx], dtype=torch.long),
            "osl": torch.tensor(self.out_tokens[idx], dtype=torch.float32),
        }


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model on a dataloader. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_pb = 0.0
    n_correct = 0
    n_total = 0
    all_preds = []
    all_targets = []
    all_q_preds = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_ids = batch["model_id"].to(device)
        dataset_ids = batch["dataset_id"].to(device)
        bucket_labels = batch["bucket_label"].to(device)
        osl = batch["osl"].to(device)

        out = model(input_ids, attention_mask, model_ids, dataset_ids)
        cls_logits = out.get("cls_logits")
        q_pred = out.get("quantiles")

        loss, components = loss_fn(cls_logits, q_pred, bucket_labels, osl)
        total_loss += components["total"] * len(osl)
        total_ce += components.get("ce", 0) * len(osl)
        total_pb += components.get("pinball", 0) * len(osl)

        if cls_logits is not None:
            preds = cls_logits.argmax(dim=1)
            n_correct += (preds == bucket_labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(bucket_labels.cpu().numpy())

        if q_pred is not None:
            all_q_preds.append(q_pred.cpu().numpy())

        n_total += len(osl)

    metrics = {
        "loss": total_loss / max(n_total, 1),
        "ce_loss": total_ce / max(n_total, 1),
        "pinball_loss": total_pb / max(n_total, 1),
    }

    if all_preds:
        metrics["accuracy"] = n_correct / max(n_total, 1)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        # Within-1-bucket accuracy
        metrics["accuracy_within_1"] = np.mean(np.abs(all_preds - all_targets) <= 1)

    if all_q_preds:
        q_preds = np.concatenate(all_q_preds, axis=0)
        metrics["q_pred_mean"] = q_preds.mean(axis=0).tolist()

    return metrics


# ── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_ids = batch["model_id"].to(device)
        dataset_ids = batch["dataset_id"].to(device)
        bucket_labels = batch["bucket_label"].to(device)
        osl = batch["osl"].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, model_ids, dataset_ids)

        loss, _ = loss_fn(
            out.get("cls_logits"), out.get("quantiles"),
            bucket_labels, osl,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(osl)
        n_total += len(osl)

    return total_loss / max(n_total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train OSL predictor")
    # Data
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to results/workload/")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model keys to include (e.g. llama3_8B)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names to include (e.g. govreport)")
    parser.add_argument("--split-by", choices=["sample", "dataset"],
                        default="sample",
                        help="Split strategy: random or hold-out-dataset")
    # Model
    parser.add_argument("--level", choices=["L2", "L3"], default="L3",
                        help="Predictor level: L2 (lightweight) or L3 (full)")
    parser.add_argument("--n-buckets", type=int, default=8,
                        help="Number of OSL classification buckets")
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 0.9],
                        help="Quantile levels for regression head")
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output-dir", type=str, default="osl_prediction/checkpoints",
                        help="Directory to save checkpoints and metrics")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs (0=best only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config: %s", json.dumps(config, indent=2))

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──
    log.info("Loading data from %s ...", args.results_dir)
    df = load_inference_results(
        args.results_dir, models=args.models, datasets=args.datasets,
    )
    log.info("Loaded %d unique samples", len(df))
    log.info("Models: %s", df["model"].unique().tolist())
    log.info("Datasets: %s", df["dataset"].unique().tolist())

    if len(df) == 0:
        log.error("No data loaded! Check --results-dir and --models/--datasets")
        return

    # ── OSL buckets ──
    _, bin_edges = create_osl_buckets(df["out_tokens"], n_buckets=args.n_buckets)
    n_classes = len(bin_edges) - 1
    log.info("OSL bucket edges: %s  (%d classes)", bin_edges.tolist(), n_classes)

    # ── Model ID mapping ──
    model_keys = sorted(df["model"].unique())
    model_to_id = {m: i for i, m in enumerate(model_keys)}
    n_models = len(model_keys)
    log.info("Model ID mapping: %s", model_to_id)

    # ── Dataset ID mapping ──
    dataset_keys = sorted(df["dataset"].unique())
    dataset_to_id = {d: i for i, d in enumerate(dataset_keys)}
    n_datasets = len(dataset_keys)
    log.info("Dataset ID mapping: %s", dataset_to_id)

    # ── Split ──
    train_df, val_df, test_df = get_split(
        df, seed=args.seed, split_by=args.split_by,
    )
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # ── Tokenizer ──
    backbone = "distilbert-base-uncased"
    max_length = 128 if args.level == "L2" else 512
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # ── Datasets ──
    train_ds = OSLDataset(train_df, tokenizer, max_length, model_to_id, dataset_to_id, bin_edges)
    val_ds = OSLDataset(val_df, tokenizer, max_length, model_to_id, dataset_to_id, bin_edges)
    test_ds = OSLDataset(test_df, tokenizer, max_length, model_to_id, dataset_to_id, bin_edges)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                             num_workers=4, pin_memory=True)

    # ── Model ──
    quantiles = tuple(args.quantiles)
    if args.level == "L2":
        model = build_l2_predictor(n_classes=n_classes, n_models=n_models,
                                   n_datasets=n_datasets, quantiles=quantiles)
    else:
        model = build_l3_predictor(n_classes=n_classes, n_models=n_models,
                                   n_datasets=n_datasets, quantiles=quantiles)
    model = model.to(device)
    log.info("Model: %s  params=%d",
             args.level, sum(p.numel() for p in model.parameters()))
    log.info("Trainable params: %d",
             sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
    )

    # ── Loss ──
    loss_fn = CombinedLoss(
        quantiles=quantiles,
        cls_weight=args.cls_weight,
        q_weight=args.q_weight,
    ).to(device)

    # ── Training loop ──
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        log.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.3f  val_acc±1=%.3f  val_pinball=%.4f  [%.1fs]",
            epoch, args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics.get("accuracy", 0),
            val_metrics.get("accuracy_within_1", 0),
            val_metrics.get("pinball_loss", 0),
            elapsed,
        )

        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
                "model_to_id": model_to_id,
                "dataset_to_id": dataset_to_id,
                "bin_edges": bin_edges.tolist(),
                "quantiles": list(quantiles),
            }, os.path.join(args.output_dir, "best_model.pt"))
            log.info("  -> Saved best model (val_loss=%.4f)", best_val_loss)

        # Periodic save
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"),
            )

    # ── Final evaluation on test set ──
    log.info("=" * 60)
    log.info("Loading best model for test evaluation...")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    log.info(
        "TEST  loss=%.4f  acc=%.3f  acc±1=%.3f  pinball=%.4f",
        test_metrics["loss"],
        test_metrics.get("accuracy", 0),
        test_metrics.get("accuracy_within_1", 0),
        test_metrics.get("pinball_loss", 0),
    )

    # ── Per-model and per-dataset breakdown ──
    log.info("Per-model/dataset test breakdown:")
    for (m, d), group in test_df.groupby(["model", "dataset"]):
        if len(group) < 5:
            continue
        sub_ds = OSLDataset(group.reset_index(drop=True), tokenizer, max_length,
                            model_to_id, dataset_to_id, bin_edges)
        sub_loader = DataLoader(sub_ds, batch_size=args.batch_size * 2,
                                num_workers=2)
        sub_metrics = evaluate(model, sub_loader, loss_fn, device)
        log.info("  %s / %s  (n=%d)  acc=%.3f  acc±1=%.3f  pinball=%.4f",
                 m, d, len(group),
                 sub_metrics.get("accuracy", 0),
                 sub_metrics.get("accuracy_within_1", 0),
                 sub_metrics.get("pinball_loss", 0))

    # ── Save results ──
    results = {
        "config": config,
        "model_to_id": model_to_id,
        "dataset_to_id": dataset_to_id,
        "bin_edges": bin_edges.tolist(),
        "history": history,
        "test_metrics": test_metrics,
        "data_stats": {
            "total_samples": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "models": model_keys,
            "datasets": df["dataset"].unique().tolist(),
        },
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s/results.json", args.output_dir)


if __name__ == "__main__":
    main()

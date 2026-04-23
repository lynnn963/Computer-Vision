from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from dataset import class_balance_stats
from eval import compute_binary_metrics, predict, save_confusion, save_metrics, save_roc_curve


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_f1: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model(model, dataloaders, split_bundle, args, device: torch.device) -> Path:
    balance = class_balance_stats(split_bundle.train_df)
    pos_weight = torch.tensor([balance["pos_weight"]], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    checkpoint_path = args.models_dir / f"{run_name}_last.pt"
    best_path = args.models_dir / f"{run_name}_best.pt"
    history_path = args.metrics_dir / f"{run_name}_history.json"

    start_epoch = 1
    best_f1 = -1.0
    patience_counter = 0
    history = {"train": [], "val": [], "class_balance": balance}

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1", -1.0))

    best_state = copy.deepcopy(model.state_dict())
    since = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels, _ in tqdm(dataloaders["train"], desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(split_bundle.train_df), 1)
        train_probs, train_labels, _ = predict(model, dataloaders["train"], device)
        val_probs, val_labels, _ = predict(model, dataloaders["val"], device)
        train_metrics = compute_binary_metrics(train_probs, train_labels, args.threshold)
        val_metrics = compute_binary_metrics(val_probs, val_labels, args.threshold)
        train_metrics["loss"] = train_loss
        val_metrics["epoch"] = epoch
        train_metrics["epoch"] = epoch
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        current_f1 = val_metrics["f1"]
        save_checkpoint(checkpoint_path, epoch, model, optimizer, best_f1, args)

        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(best_path, epoch, model, optimizer, best_f1, args)
            save_metrics(val_metrics, args.metrics_dir / f"{run_name}_best_val_metrics.json")
            save_roc_curve(val_labels, val_probs, args.figures_dir / f"{run_name}_val_roc.png")
            save_confusion(val_labels, val_probs, args.threshold, args.figures_dir / f"{run_name}_val_confusion.png")
        else:
            patience_counter += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - since
    print(f"Training completed in {elapsed / 60:.1f} minutes.")
    return best_path

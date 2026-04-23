from __future__ import annotations

import random

import numpy as np
import torch

from config import build_parser, make_run_name
from dataset import make_dataloaders
from eval import compute_binary_metrics, predict, save_confusion, save_metrics, save_roc_curve
from gradcam import generate_gradcam_examples
from model import create_model
from train import load_checkpoint, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_output_dirs(args) -> None:
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)


def evaluate_split(model, dataloader, args, device: torch.device, split_name: str) -> dict:
    probabilities, labels, _ = predict(model, dataloader, device)
    metrics = compute_binary_metrics(probabilities, labels, args.threshold)
    save_metrics(metrics, args.metrics_dir / f"{args.run_name}_{split_name}_metrics.json")
    save_roc_curve(labels, probabilities, args.figures_dir / f"{args.run_name}_{split_name}_roc.png")
    save_confusion(labels, probabilities, args.threshold, args.figures_dir / f"{args.run_name}_{split_name}_confusion.png")
    print(f"{split_name} metrics: {metrics}")
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.run_name = make_run_name(args)
    ensure_output_dirs(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataloaders, sizes, split_bundle = make_dataloaders(args)
    print(f"Dataset sizes: {sizes}")
    model = create_model(args.backbone, args.pretrained, args.dropout).to(device)

    if args.mode == "train":
        best_path = train_model(model, dataloaders, split_bundle, args, device)
        load_checkpoint(best_path, model)
        evaluate_split(model, dataloaders["test"], args, device, "test")
        return

    checkpoint_path = args.checkpoint or args.resume
    if checkpoint_path is None:
        raise ValueError("`--checkpoint` is required for eval and gradcam modes.")
    load_checkpoint(checkpoint_path, model)

    if args.mode == "eval":
        evaluate_split(model, dataloaders["test"], args, device, "test")
        return

    if args.mode == "gradcam":
        generate_gradcam_examples(model, dataloaders["test"], args, device)
        print(f"Grad-CAM figures saved to {args.figures_dir / 'gradcam'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary chest X-ray abnormality detection")

    parser.add_argument("--mode", choices=["train", "eval", "gradcam"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DATA_DIR / "Data_Entry_2017.csv")
    parser.add_argument("--trainval-list", type=Path, default=DATA_DIR / "train_val_list_NIH.txt")
    parser.add_argument("--test-list", type=Path, default=DATA_DIR / "test_list_NIH.txt")
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR / "images-224" / "images-224")
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)

    parser.add_argument("--backbone", choices=["resnet18", "densenet121"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gradcam-count", type=int, default=8)

    parser.add_argument("--outputs-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=OUTPUT_DIR / "models")
    parser.add_argument("--metrics-dir", type=Path, default=OUTPUT_DIR / "metrics")
    parser.add_argument("--figures-dir", type=Path, default=OUTPUT_DIR / "figures")

    return parser


def make_run_name(args: argparse.Namespace) -> str:
    aug = "aug" if args.augment else "noaug"
    pre = "pretrained" if args.pretrained else "scratch"
    return f"{args.backbone}_{pre}_{aug}_{args.img_size}px_bs{args.batch_size}"

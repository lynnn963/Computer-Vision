from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class NIHXrayBinaryDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.image_dir / row["image_name"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return image, label, row["image_name"]


def build_transforms(img_size: int, augment: bool) -> Dict[str, transforms.Compose]:
    train_transforms: List[transforms.Compose] = [transforms.Resize((img_size, img_size))]
    if augment:
        train_transforms.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
            ]
        )

    tail = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {
        "train": transforms.Compose(train_transforms + tail),
        "eval": eval_transform,
    }


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_csv, usecols=["Image Index", "Finding Labels", "Patient ID"])
    frame = frame.rename(columns={"Image Index": "image_name", "Finding Labels": "finding_labels", "Patient ID": "patient_id"})
    frame["label"] = (frame["finding_labels"] != "No Finding").astype(int)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def read_split_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_split_frames(
    metadata_csv: Path,
    trainval_list: Path,
    test_list: Path,
    val_ratio: float,
    seed: int,
) -> SplitBundle:
    metadata = load_metadata(metadata_csv)
    trainval_names = set(read_split_file(trainval_list))
    test_names = set(read_split_file(test_list))

    trainval_df = metadata[metadata["image_name"].isin(trainval_names)].copy()
    test_df = metadata[metadata["image_name"].isin(test_names)].copy()

    patient_frame = (
        trainval_df.groupby("patient_id", as_index=False)["label"]
        .max()
        .rename(columns={"label": "patient_label"})
    )
    train_patients, val_patients = train_test_split(
        patient_frame["patient_id"],
        test_size=val_ratio,
        random_state=seed,
        stratify=patient_frame["patient_label"],
    )

    train_df = trainval_df[trainval_df["patient_id"].isin(set(train_patients))].copy()
    val_df = trainval_df[trainval_df["patient_id"].isin(set(val_patients))].copy()

    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df)


def make_dataloaders(args) -> tuple[Dict[str, DataLoader], Dict[str, int], SplitBundle]:
    split_bundle = build_split_frames(
        metadata_csv=args.metadata_csv,
        trainval_list=args.trainval_list,
        test_list=args.test_list,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    tfms = build_transforms(args.img_size, args.augment)

    datasets = {
        "train": NIHXrayBinaryDataset(split_bundle.train_df, args.image_dir, tfms["train"]),
        "val": NIHXrayBinaryDataset(split_bundle.val_df, args.image_dir, tfms["eval"]),
        "test": NIHXrayBinaryDataset(split_bundle.test_df, args.image_dir, tfms["eval"]),
    }
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    sizes = {split: len(dataset) for split, dataset in datasets.items()}
    return dataloaders, sizes, split_bundle


def class_balance_stats(train_df: pd.DataFrame) -> dict:
    positives = int(train_df["label"].sum())
    negatives = int(len(train_df) - positives)
    pos_weight = float(negatives / positives) if positives else 1.0
    return {
        "train_samples": int(len(train_df)),
        "train_positives": positives,
        "train_negatives": negatives,
        "pos_weight": pos_weight,
    }

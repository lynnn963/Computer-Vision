from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@torch.no_grad()
def predict(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probabilities = []
    labels = []
    image_names: list[str] = []

    for images, batch_labels, batch_names in dataloader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        probabilities.extend(probs.tolist())
        labels.extend(batch_labels.squeeze(1).cpu().numpy().tolist())
        image_names.extend(list(batch_names))

    return np.asarray(probabilities), np.asarray(labels), image_names


def compute_binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_roc_curve(labels: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fpr, tpr, _ = roc_curve(labels, probabilities)
    except ValueError:
        return

    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion(labels: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

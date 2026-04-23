from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(backbone: str, pretrained: bool, dropout: float) -> nn.Module:
    weights = None
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        return model

    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def gradcam_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    if backbone == "resnet18":
        return model.layer4[-1]
    if backbone == "densenet121":
        return model.features[-1]
    raise ValueError(f"Unsupported backbone: {backbone}")

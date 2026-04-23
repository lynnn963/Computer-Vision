from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import gradcam_target_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits.squeeze()
        score.backward(retain_graph=True)

        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (gradients * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(weighted).squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (image_tensor.shape[-1], image_tensor.shape[-2]))
        heatmap = heatmap - heatmap.min()
        denom = heatmap.max() if heatmap.max() > 0 else 1.0
        return heatmap / denom


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def generate_gradcam_examples(model, dataloader, args, device: torch.device) -> None:
    model.eval()
    target = gradcam_target_layer(model, args.backbone)
    cam = GradCAM(model, target)
    save_dir = args.figures_dir / "gradcam"
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for images, labels, names in dataloader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= args.threshold).astype(int)

        for index in range(images.size(0)):
            image_tensor = images[index : index + 1]
            heatmap = cam(image_tensor)
            image = denormalize(images[index])
            overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            blended = np.clip(0.6 * image + 0.4 * overlay, 0, 255).astype(np.uint8)

            name = Path(names[index]).stem
            label = int(labels[index].item())
            pred = int(preds[index])
            prob = float(probs[index])
            Image.fromarray(blended).save(save_dir / f"{name}_label{label}_pred{pred}_p{prob:.3f}.png")
            saved += 1
            if saved >= args.gradcam_count:
                cam.remove()
                return

    cam.remove()

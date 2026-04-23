# Abnormality Detection

Binary chest X-ray classification pipeline based on the NIH ChestX-ray14 metadata already placed in `data/`.

## Task

- `0`: normal (`No Finding`)
- `1`: abnormal (any pathology label)

## Structure

- `data/`: local dataset files, ignored by Git
- `src/`: training, evaluation, and Grad-CAM pipeline
- `outputs/models/`: checkpoints
- `outputs/metrics/`: JSON metrics and training history
- `outputs/figures/`: ROC curves, confusion matrices, Grad-CAM examples

## Run

From the repository root:

```powershell
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py --mode train --backbone resnet18 --epochs 10 --batch-size 32
```

Evaluate a saved checkpoint:

```powershell
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py --mode eval --checkpoint abnormality_detection\outputs\models\resnet18_pretrained_aug_224px_bs32_best.pt
```

Generate Grad-CAM examples:

```powershell
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py --mode gradcam --checkpoint abnormality_detection\outputs\models\resnet18_pretrained_aug_224px_bs32_best.pt --gradcam-count 8
```

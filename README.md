# Chest X-Ray Abnormality Detection

Binary chest X-ray classification using transfer learning. Each image is classified as **normal** (no pathology) or **abnormal** (any pathology), with Grad-CAM visualizations to explain model decisions.

**Team:** Jose Azzi · Ilona Chamoun · Lynn Mechreck  
**Course:** Computer Vision — Saint Joseph University, Spring 2026  
**Supervisor:** Dr. Ahmad Audi

---

## Task

Labels from the NIH ChestX-ray14 dataset are collapsed into a binary target:

| Label | Meaning |
|---|---|
| `0` | Normal — original label is `No Finding` |
| `1` | Abnormal — any pathology label |

---

## Dataset

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — a sample subset of 5,606 images from 4,230 patients.

| Split | Images | Patients | Abnormal % |
|---|---|---|---|
| Train | 4,012 | 3,055 | 45.54% |
| Val | 750 | 540 | 45.07% |
| Test | 844 | 635 | 47.04% |

Splits are done at **patient level** to prevent leakage. Download the dataset and place it at:

```
abnormality_detection/data/sample/
  sample_labels.csv
  images/
```

---

## Setup

```bash
cd abnormality_detection
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running

All commands are run from the **repository root**.

### Train

```bash
# Baseline: ResNet18 with augmentation (best model)
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py ^
  --mode train --backbone resnet18 --epochs 5 --batch-size 8 --augment

# Ablation: ResNet18 without augmentation
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py ^
  --mode train --backbone resnet18 --epochs 5 --batch-size 8 --no-augment

# Architecture comparison: DenseNet121 with augmentation
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py ^
  --mode train --backbone densenet121 --epochs 5 --batch-size 8 --augment
```

Training saves checkpoints to `abnormality_detection/outputs/models/` and metrics to `outputs/metrics/`.

### Evaluate (inference on test set)

```bash
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py ^
  --mode eval ^
  --checkpoint abnormality_detection\outputs\models\resnet18_pretrained_aug_224px_bs8_best.pt
```

### Generate Grad-CAM visualizations

```bash
abnormality_detection\venv\Scripts\python.exe abnormality_detection\src\main.py ^
  --mode gradcam ^
  --checkpoint abnormality_detection\outputs\models\resnet18_pretrained_aug_224px_bs8_best.pt ^
  --gradcam-count 8
```

---

## Reproducing Training

Trained weights are not stored in this repository (files are 80–130 MB each). To reproduce the experiments, run the three training commands above in order. Each run takes approximately 10–15 minutes on CPU or 2–3 minutes on GPU. The best checkpoint is automatically saved as `*_best.pt` based on validation F1-score.

---

## Results

Three experiments were run to compare architectures and the effect of augmentation.

| Run | Backbone | Augmentation | Best Epoch | Val F1 | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline | ResNet18 | Yes | 4 | 0.6842 | 0.6611 | 0.7159 |
| Ablation | ResNet18 | No | 3 | 0.6765 | 0.6712 | 0.7088 |
| Comparison | DenseNet121 | Yes | 1 | 0.6599 | 0.6581 | 0.7149 |

**Selected model:** ResNet18 with augmentation (highest validation F1).

### Test ROC curve

![Test ROC curve](abnormality_detection/outputs/figures/resnet18_pretrained_aug_224px_bs8_test_roc.png)

### Test confusion matrix

![Test confusion matrix](abnormality_detection/outputs/figures/resnet18_pretrained_aug_224px_bs8_test_confusion.png)

### Grad-CAM examples

| Correct abnormal | Correct abnormal |
|---|---|
| ![](abnormality_detection/outputs/figures/gradcam/00000013_005_label1_pred1_p0.759.png) | ![](abnormality_detection/outputs/figures/gradcam/00000096_006_label1_pred1_p0.800.png) |

| Correct normal | False negative (missed case) |
|---|---|
| ![](abnormality_detection/outputs/figures/gradcam/00000042_002_label0_pred0_p0.304.png) | ![](abnormality_detection/outputs/figures/gradcam/00000030_001_label1_pred0_p0.496.png) |

---

## Repository Structure

```
abnormality_detection/
  src/
    config.py      argument parser and run-name builder
    dataset.py     data loading, label collapsing, patient-level splits
    model.py       ResNet18 / DenseNet121 with binary head
    train.py       training loop, early stopping, checkpointing
    eval.py        metrics, ROC curves, confusion matrices
    gradcam.py     Grad-CAM heatmap generation
    main.py        entry point for train / eval / gradcam modes
  requirements.txt
  outputs/
    models/        saved checkpoints (not tracked by git)
    metrics/       per-epoch and final JSON metrics
    figures/       ROC curves, confusion matrices, Grad-CAM images
```

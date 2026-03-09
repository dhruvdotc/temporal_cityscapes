# temporal_cityscapes

Robust semantic segmentation under visual corruption using confidence-based gating and reconstruction on the Cityscapes dataset.

This project investigates how segmentation models behave when input images are degraded (blur, noise, brightness shifts), and proposes a confidence-aware reconstruction pipeline to improve reliability of predictions.

Final project for a computer vision / machine learning course.

---

# Project Motivation

Autonomous systems rely heavily on semantic segmentation to interpret their environment. However, real-world conditions often degrade image quality through:

- motion blur  
- sensor noise  
- lighting variation  

These corruptions can cause segmentation models to produce low-confidence or incorrect predictions.

This project studies:

1. How segmentation accuracy degrades under common visual distortions.
2. Whether confidence gating can identify unreliable pixels.
3. Whether reconstruction methods can recover segmentation quality.

---

# Method Overview

The pipeline consists of four stages.

## 1. Baseline Training

A semantic segmentation model is trained on clean Cityscapes images.

Notebook:

notebooks/01_baseline_train.ipynb

Outputs include:

- training curves
- baseline metrics
- trained model weights

---

## 2. Corruption Evaluation

The trained model is evaluated on corrupted images.

Corruptions tested:

- Gaussian blur
- noise
- brightness shift
- contrast shift

Notebook:

notebooks/02_shift_eval.ipynb

Metrics computed:

- mIoU
- pixel accuracy
- boundary F1 score

Results are saved to:

outputs/shift_eval_results.csv

---

## 3. Confidence-Based Gating

The model’s softmax confidence is used to detect unreliable predictions.

Pixels below a confidence threshold are marked as uncertain.

Notebook:

notebooks/03_confidence_gating.ipynb

Generated outputs include:

- confidence heatmaps
- gating masks
- confidence vs accuracy plots

Example outputs:

outputs/gating_mask.png  
outputs/confidence_heatmap.png

---

## 4. Reconstruction of Uncertain Regions

Low-confidence regions are reconstructed using post-processing methods such as:

- nearest-neighbor propagation
- region-based reconstruction

Notebook:

notebooks/04_postprocess_reconstruction.ipynb

Outputs include:

outputs/reconstruction_results.txt  
outputs/qual_recon_blur.png

---

# Key Results

Main evaluation metrics:

| Method | mIoU |
|------|------|
| Baseline corrupted input | ~0.408 |
| Confidence-gated reconstruction | ~0.435 |

Additional improvements observed in:

- boundary F1 score
- pixel accuracy in corrupted regions

Full metrics are available in:

outputs/summary_metrics.csv  
outputs/metrics_miou.csv  
outputs/metrics_pixel_acc.csv

---

# Repository Structure
```text
temporal_cityscapes/
├── notebooks/
│       └── pdf/
│           ├── 01_baseline_train.pdf
│           ├── 02_shift_eval.pdf
│           ├── 03_confidence_gating.pdf
│           └── 04_postprocess_reconstruction.pdf
│       └── raw/
│           ├── 01_baseline_train.ipynb
│           ├── 02_shift_eval.ipynb
│           ├── 03_confidence_gating.ipynb
│           └── 04_postprocess_reconstruction.ipynb
├── outputs/
│       └── [contains several .txt & .csv figures explored in the report]
└── src/
│       ├── dataset_cityscapes.py
│       ├── metrics.py
│       └── train_utils.py
```


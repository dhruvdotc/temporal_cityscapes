# temporal_cityscapes

Robust semantic segmentation under visual corruption using confidence-based gating and reconstruction on the Cityscapes dataset.

This project investigates how segmentation models behave when input images are degraded (blur, noise, brightness shifts), and proposes a confidence-aware reconstruction pipeline to improve reliability of predictions.

---

# Report

The full project report is included in the repository as:

[Project Report (PDF)](Project_Report___ECE_176__1_.pdf)

or view it directly via GitHub Pages:

[Standalone report link](https://dhruvdotc.github.io/temporal_cityscapes/Project_Report___ECE_176__1_.pdf)

The report contains:

- theoretical background
- experimental methodology
- evaluation results
- discussion and future work

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

Main evaluation metrics (averaged across distortions):

| Setting | mIoU |
|-------|------|
| Corrupted baseline | ~0.387 |
| After confidence-guided reconstruction | ~0.427 |

Overall improvement:

- **+0.039 mIoU recovery** over the corrupted baseline

These results indicate that lightweight post-processing methods can partially recover segmentation accuracy under distribution shift without retraining the model.

Additional evaluation metrics include:

- pixel accuracy
- boundary F1 score
- change-focused diagnostics (net gain and error reduction)

Key figures and plots are available in:

outputs/miou_grouped_by_distortion.png  
outputs/pixel_accuracy_grouped_by_distortion.png  
outputs/boundary_f1_grouped_by_distortion.png  

Full metrics tables:

outputs/summary_metrics_table.csv  
outputs/metrics_miou.csv  
outputs/metrics_pixel_accuracy.csv  
outputs/metrics_boundary_f1.csv

---

# Repository Structure
```text
temporal_cityscapes/
├── notebooks/
│       ├── pdf/
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
│       ├── boundary_f1_grouped_by_distortion.png
│       ├── . . .
│       └── training_curve.png
├── src/
│       ├── dataset_cityscapes.py
│       ├── metrics.py
│       └── train_utils.py
```


---

# Dataset

This project uses the **Cityscapes dataset**.

Due to its large size, the dataset is **not included in the repository**.

Download from:

https://www.cityscapes-dataset.com/

Place the dataset in the following structure:
```text
temporal_cityscapes/
├── data/
│       ├── gtFine/
│           └── ...
│       └── leftImg8bit/
│           └── ...
```   


---

# Setup

Recommended Python version:

Python 3.9+

Install dependencies:
```text
pip install torch torchvision numpy matplotlib scikit-image pandas
```


---

# Running the Project

Run the notebooks in the following order:

1. notebooks/01_baseline_train.ipynb  
2. notebooks/02_shift_eval.ipynb  
3. notebooks/03_confidence_gating.ipynb  
4. notebooks/04_postprocess_reconstruction.ipynb  

Each notebook saves outputs to the `outputs/` directory.

---

# Visual Results

Example outputs generated by the pipeline include:

- segmentation predictions
- confidence heatmaps
- gated uncertainty regions
- reconstructed segmentation maps

All figures are saved in:
```text
├── outputs/
│       ├── boundary_f1_grouped_by_distortion.png
│       ├── . . .
│       └── training_curve.png

```

---

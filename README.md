# Windshield Detection — YOLOv8

**Object detection · Computer vision · Deep learning**

This project focuses on automatic windshield detection in vehicle images using a deep learning pipeline based on YOLOv8.

The objective is to localize windshields with high precision while ensuring that the reported performance is reliable, reproducible, and statistically validated.

Instead of training a single model with arbitrary hyperparameters, the project follows a rigorous two-stage methodology:

1. Hyperparameter optimization using **Grid Search + 3-Fold Cross-Validation**
2. Final training and unbiased evaluation on a **held-out test set**

The entire pipeline was developed and executed on **Kaggle** using a **Tesla T4 GPU**.

---

## Project Overview

| | |
|---|---|
| **Task** | Object detection (single class: windshield) |
| **Dataset** | 556 annotated images |
| **Final mAP50** | 0.967 on held-out test set |
| **Framework** | YOLOv8 + PyTorch |
| **Hardware** | Tesla T4 GPU |

---

## Project Highlights

- 556 annotated images
- 64 hyperparameter configurations tested
- 192 total training runs (64 × 3 folds)
- Statistical validation using the **Friedman test**
- Final unbiased evaluation on a held-out test set
- **Final mAP50 = 0.967**
- Inference speed ≈ **6 ms/image**

---

## Dataset

The dataset contains vehicle images annotated for a single class:

```
Class: windshield
```

### Dataset Split

| Split | Images | Role |
|---|---|---|
| Train | 390 | Model training |
| Validation | 103 | Early stopping & overfitting monitoring |
| Test | 59 | Final evaluation only |

> The test set was **never used** during training, hyperparameter selection, or cross-validation.
> This guarantees a fully unbiased final evaluation.

---

## Methodology

The project is divided into two notebooks.

### Notebook 1 — Grid Search + K-Fold Cross-Validation

This notebook performs a systematic search over multiple YOLOv8 hyperparameters.

#### Search Space

| Hyperparameter | Values |
|---|---|
| Model | `yolov8n`, `yolov8s` |
| Optimizer | `SGD`, `AdamW` |
| Learning rate (lr0) | `0.001`, `0.005` |
| Weight decay | `0.0005`, `0.001` |
| Image size | `512`, `640` |
| Batch size | `8`, `16` |

#### Training Setup

- 3-Fold Cross-Validation
- 40 epochs max
- Early stopping patience = 12
- Seed = 42

#### Total Runs

**64 configurations × 3 folds = 192 runs**

#### Best Configuration Found

| Parameter | Value |
|---|---|
| Model | `yolov8s.pt` |
| Optimizer | `AdamW` |
| Learning rate | `0.001` |
| Weight decay | `0.001` |
| Batch size | `8` |
| Image size | `512` |

#### Cross-Validation Performance

| Metric | Value |
|---|---|
| mAP50 | 0.9827 |
| mAP50-95 | 0.7026  |
| Precision | 0.9710  |
| Recall | 0.9457  |

#### Statistical Analysis

A Friedman statistical test was performed across all evaluated configurations:

- **χ² = 57.35**
- **p = 0.677**

The result indicates that the observed performance differences are **not statistically significant**. This suggests that the task is relatively simple and that several configurations perform similarly. The Friedman test confirms that simpler models can achieve performance comparable to larger ones on this dataset.

---

### Notebook 2 — Final Training & Evaluation

Although `yolov8s` achieved the highest average score during Grid Search, the Friedman test showed that the performance gap was not statistically significant.

For this reason, **`yolov8n` was selected** for final deployment because it is:

- Lighter (3M vs 11M parameters)
- Faster (8.2 vs 28.4 GFLOPs)
- Less computationally expensive
- Less prone to overfitting on a small dataset


#### Final Configuration

| Parameter | Value |
|---|---|
| Model | `yolov8n.pt` |
| Optimizer | `AdamW` |
| Learning rate | `0.001` |
| Weight decay | `0.001` |
| Batch size | `8` |
| Image size | `512` |

#### Final Test Results

Evaluation was performed **once** on the held-out test set.

| Metric | Value |
|---|---|
| **mAP50** | **0.967** |
| mAP50-95 | 0.615 |
| Precision | 0.954 |
| Recall | 0.942 |
| Inference speed | ~6 ms/image |

> The small drop between cross-validation and test performance is expected and indicates good generalization without significant overfitting.

---

## Repository Structure

```
windshield-detection/
│
├── README.md
│
├── notebooks/
│   ├── 01_grid_search_windshield_detection.ipynb
│   └── 02_training_windshield_detection.ipynb
│
├── results/
│   │
│   ├── notebook1_gridsearch/
│   │   ├── best_config.json
│   │   ├── gridsearch_raw.csv
│   │   ├── gridsearch_aggregated.csv
│   │   └── figures/
│   │       ├── heatmap_mAP50.png
│   │       ├── bar_mAP50.png
│   │       ├── boxplot_top5.png
│   │       └── marginal_impact.png
│   │
│   └── notebook2_training/
│       ├── results.csv
│       └── figures/
│           ├── training_curves.png
│           ├── confusion_matrix.png
│           └── cv_vs_test_comparison.png
│           └── results.png
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/farahGaida/part1-windshield-detection.git
cd windshield-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Expected dataset structure:

```
dataset_windshield/
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

Labels follow YOLO format:

```
class x_center y_center width height
```

---

## How to Run

Run the notebooks **in order**.

### 1. Grid Search + Cross-Validation
`01_grid_search_windshield_detection.ipynb`

This notebook:
- Performs hyperparameter optimization
- Runs 3-fold cross-validation
- Performs statistical analysis (Friedman test)
- Saves `best_config.json`

### 2. Final Training & Evaluation
`02_training_windshield_detection.ipynb`

This notebook:
- Retrains the selected model using `best_config.json`
- Monitors validation performance (early stopping)
- Evaluates once on the held-out test set
- Generates final figures and metrics

---

## Requirements

```
ultralytics>=8.4.0
pandas
numpy
matplotlib
seaborn
scipy
pyyaml
scikit-learn
```

---

## Tech Stack

- Python 3.12
- Ultralytics YOLOv8
- PyTorch + CUDA / Tesla T4 GPU
- Pandas · NumPy · Matplotlib
- SciPy · Scikit-learn

---

## Key Design Decisions

**Why use K-Fold Cross-Validation?**
With only 556 images, a single validation split may produce unstable estimates. 3-fold cross-validation provides a more reliable and robust hyperparameter selection process.

**Why keep train and validation sets separate in Notebook 2?**
Monitoring validation metrics during training is important to detect overfitting and correctly apply early stopping.

**Why switch from `yolov8s` to `yolov8n`?**
The Friedman test confirmed that the performance difference is not statistically significant. Therefore, `yolov8n` offers a better efficiency/performance trade-off.

**Why use the test set only once?**
Using the test set multiple times would introduce data leakage and produce biased results. The test set is reserved exclusively for the final evaluation.

---


# Vision-Based Industrial Anomaly Detection

Computer vision project implementing a PatchCore-style anomaly detection pipeline using **PatchCore** with multi-scale feature extraction, hybrid scoring, and anomaly heatmap localization. Built for the **MVTec AD** benchmark as a BSc final year project.

## Evaluation targets

The values below are project goals, not measured benchmark results. Report achieved metrics separately with the dataset split, configuration, and evaluation artifacts so results can be reproduced.

| Metric | Target |
| --- | --- |
| Mean AUROC (image-level) | ≥ 0.90 |
| Bottle AUROC | ≥ 0.90 |
| Cable AUROC | ≥ 0.92 |
| Capsule AUROC | ≥ 0.88 |

## 🔬 Method Overview

```
Input Image → ResNet-50 Backbone → Multi-Scale Feature Maps (layer2 + layer3)
  → Patch Embeddings → PCA Compression → Coreset Subsampling
  → Patch Memory Bank → kNN Anomaly Scoring → Heatmap Localization
  → Score Calibration → Confidence (0–100%)
```

### Architecture

- **Backbone**: ResNet-50 (ImageNet pretrained) — no training required
- **Feature Extraction**: PatchCore-style multi-scale spatial features from layer2 (28×28) and layer3 (14×14)
- **Memory Bank**: Patch-wise memory of normal features with greedy k-center coreset subsampling (10%)
- **Scoring**: kNN distance to nearest patches (primary), optional Mahalanobis + cosine ensemble
- **Localization**: Pixel-level anomaly heatmaps via patch score upsampling + Gaussian smoothing
- **Calibration**: Min-max + sigmoid calibration for interpretable 0–100% confidence scores

### Alternative Pipeline (Ablation)

- SimCLR contrastive pretraining (optional)
- Global multi-layer feature extraction (layer2 + layer3 + layer4 → 3584-dim)
- Mahalanobis distance scoring
- Grad-CAM localization

## 📁 Project Structure

```
BSC_PROJECT/
├── configs/
│   └── config.yaml          # All configuration (model, training, scoring, etc.)
├── scripts/
│   ├── train.py              # Training/feature extraction pipeline
│   ├── evaluate.py           # Full evaluation with metrics + plots
│   ├── inference.py          # Single-image / batch inference
│   └── download_dataset.py   # Download MVTec AD from Kaggle
├── src/
│   ├── models/
│   │   └── simclr.py         # ResNet encoder + patch feature extraction
│   ├── training/
│   │   ├── trainer.py        # SimCLR training loop with early stopping
│   │   ├── dataset.py        # MVTec AD dataset + dataloaders
│   │   ├── augmentations.py  # SimCLR augmentation pipeline
│   │   └── losses.py         # NT-Xent contrastive loss
│   ├── memory/
│   │   └── memory_bank.py    # Global + Patch memory banks + scorers
│   ├── scoring/
│   │   └── calibration.py    # Score calibration (min-max, sigmoid, temperature)
│   ├── evaluation/
│   │   └── evaluator.py      # AUROC, AP, F1, PRO, cross-category stats
│   ├── inference/
│   │   ├── predictor.py      # Production inference API (Patch + Global)
│   │   └── gradcam.py        # Grad-CAM localization (legacy)
│   ├── visualization/
│   │   └── heatmap.py        # Anomaly heatmap generation + overlay
│   └── utils/
│       └── utils.py          # Config, logging, checkpoints, seeds
├── app/
│   └── streamlit_app.py      # Interactive web interface
├── tests/
│   ├── test_unit.py
│   └── test_integration.py
├── Data/
│   └── mvtec_ad/             # MVTec AD dataset (downloaded)
├── outputs/
│   ├── checkpoints/          # Model weights + memory banks
│   ├── results/              # Metrics JSON + plots
│   ├── logs/                 # Training logs + TensorBoard
│   └── visualizations/       # Heatmap images
└── requirements.txt
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: Fast kNN with Faiss (10x speedup for PatchCore)
pip install faiss-cpu
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

Or manually download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) to `Data/mvtec_ad/`.

## 📖 Usage

### Training (Feature Extraction + Memory Bank Building)

```bash
# Single category (PatchCore pipeline — default, recommended):
python scripts/train.py --category bottle

# All 15 categories:
python scripts/train.py

# With SimCLR contrastive training (optional, slower):
python scripts/train.py --category bottle --train-simclr --epochs 50

# Legacy global feature pipeline:
python scripts/train.py --category bottle --no-patch
```

### Evaluation

```bash
# Single category:
python scripts/evaluate.py --category bottle

# All categories with cross-category summary:
python scripts/evaluate.py
```

Outputs:
- Metrics JSON (`outputs/results/{category}_metrics.json`)
- ROC curves, PR curves, F1-vs-threshold plots, confusion matrices
- Anomaly heatmap visualizations
- Cross-category summary with mean ± std

### Inference

```bash
# Single image:
python scripts/inference.py --image path/to/image.png --category bottle

# Batch (directory):
python scripts/inference.py --image_dir path/to/images/ --category bottle
```

### Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Features:
- Upload image and get anomaly prediction
- Anomaly heatmap localization
- Calibrated confidence score (0–100%)
- Performance dashboard

## ⚙️ Configuration

All parameters in `configs/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `model` | `backbone`, `pretrained`, `patch_layers` |
| `patch_detection` | `enabled`, `coreset_ratio`, `pca_components` |
| `scoring` | `k_neighbors`, `weight_knn`, `weight_mahalanobis` |
| `calibration` | `method` (minmax_sigmoid), `temperature` |
| `localization` | `gaussian_sigma`, `colormap`, `overlay_alpha` |
| `training` | `epochs`, `batch_size`, `gradient_accumulation_steps` |

## 📊 Evaluation Metrics

- **AUROC** — Image-level area under ROC curve
- **Average Precision (AP)** — Area under precision-recall curve
- **F1-Score** — Harmonic mean of precision and recall (optimal + fixed thresholds)
- **PRO Score** — Per-Region Overlap for localization quality
- **Pixel-AUROC** — Pixel-level AUROC (when GT masks available)

## 🔧 Ablation Studies

Toggle components via config or CLI to compare:
- PatchCore vs Global features
- kNN vs Mahalanobis scoring
- PCA vs no PCA
- Different feature layers (layer2, layer3, layer4)
- Different coreset ratios (1%, 10%, 25%)

Results are automatically logged to `outputs/results/ablation_results.json`.

## 📚 References

1. **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
2. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020
3. **MVTec AD**: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset", CVPR 2019

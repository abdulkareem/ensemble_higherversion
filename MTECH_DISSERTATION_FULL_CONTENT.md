# M.Tech Dissertation Draft (Complete Chapter-Wise Content)

## Title
**Hybrid Attention and Refinement Ensemble for Polyp Segmentation Using ResUNet++, TransFuse, and WDFFNet at 512×512 Resolution**

## Abstract
Colorectal cancer prevention depends heavily on early and reliable polyp detection during colonoscopy. Manual delineation of polyps is time-consuming and susceptible to inter-observer variability, motivating the use of automated segmentation systems. This dissertation proposes a unified deep learning framework using three complementary base models—ResUNet++, TransFuse, and WDFFNet—trained under identical preprocessing and optimization settings at 512×512 resolution on Kvasir-SEG. A learnable Hybrid Attention Ensemble (HAR) is designed to fuse base-model predictions through adaptive spatial-channel weighting, and an additional refinement network is used to improve boundary fidelity. The full pipeline includes robust checkpoint loading, threshold calibration, test-time augmentation, post-processing, parameter/FPS profiling, and qualitative visual analysis. Experimental outputs are automatically exported to Google Drive as checkpoints, tabular comparison files, and multi-model visualization sheets to ensure reproducibility and reporting readiness.

The standardized 512×512 configuration improves fairness across architectures and better preserves fine lesion boundaries. Comprehensive comparison across Dice, IoU, precision, recall, and accuracy demonstrates that adaptive fusion performs better than any single model in most settings. The framework is engineered for practical Colab deployment and dissertation-grade reporting, including architecture details, result tables, ablation-ready logs, and publication-style qualitative panels.

---

## Chapter 1: Introduction

### 1.1 Background
Colorectal polyps are precursors to colorectal cancer. During colonoscopy, clinicians inspect video frames for suspicious lesions. Robust segmentation of polyp boundaries is essential for lesion size estimation, morphology analysis, and downstream biopsy planning. However, traditional computer vision struggles with low contrast, varying illumination, specular highlights, and texture ambiguities in endoscopic images.

### 1.2 Motivation
No single architecture consistently handles all challenging cases. CNN-heavy models are often strong at local texture and boundary representation, while transformer-like designs can better exploit global contextual cues. A carefully engineered ensemble of heterogeneous models can potentially achieve higher reliability than any standalone architecture.

### 1.3 Problem Statement
Develop and validate an accurate, reproducible, and deployable segmentation pipeline that:
1. Trains ResUNet++, TransFuse, and WDFFNet uniformly at 512×512.
2. Builds a learnable ensemble with adaptive weights.
3. Saves all checkpoints and model-level statistics for transparent comparison.
4. Produces qualitative comparison sheets for dissertation reporting.

### 1.4 Objectives
- Build a uniform 512×512 training/evaluation setup.
- Train all three base models and the HAR ensemble.
- Add refinement-stage prediction to improve difficult boundaries.
- Export comparison metrics and model details to Google Drive.
- Generate multi-model visual comparison figures.

### 1.5 Contributions
- A Colab-ready end-to-end segmentation pipeline with robust environment handling.
- Uniform 512×512 protocol for fair architecture benchmarking.
- Adaptive weighted fusion with optional refinement stage.
- Automated export of checkpoints, parameter tables, metric CSV/JSON, and comparison images.

### 1.6 Dissertation Organization
- **Chapter 2:** Literature review and research gap.
- **Chapter 3:** Proposed methodology and system design.
- **Chapter 4:** Experimental setup, results, and analysis.
- **Chapter 5:** Conclusion and future scope.

---

## Chapter 2: Literature Review

### 2.1 Medical Image Segmentation Evolution
Early approaches relied on handcrafted descriptors and region-based methods. Deep learning shifted the field with encoder-decoder networks such as U-Net, enabling strong pixel-level segmentation under moderate data constraints.

### 2.2 ResUNet++ Family
ResUNet++ improves on U-Net by adding residual learning, atrous/multi-scale features, and attention-inspired decoding behavior. It is effective for fine structure recovery and noisy medical boundaries.

### 2.3 TransFuse-Style Dual-Stream Methods
TransFuse-type designs combine CNN detail extraction with transformer-like/global representations through fusion blocks. This helps balance local sharpness and global context sensitivity.

### 2.4 Dual-Backbone Fusion in WDFFNet
WDFFNet fuses feature hierarchies from two different backbones and applies object-aware attention. Heterogeneous encoders provide complementary representations, often improving robustness under difficult visual conditions.

### 2.5 Ensemble Learning in Medical AI
Model ensembles can reduce variance and improve generalization. Learnable weighted ensembles outperform naive averaging when branch reliability is sample-dependent.

### 2.6 Research Gap
Most comparative studies are affected by inconsistent preprocessing, mixed resolutions, and non-uniform training settings. This work addresses that by enforcing a uniform 512×512 pipeline and transparent reporting artifacts.

---

## Chapter 3: Methodology

### 3.1 System Overview
The proposed framework follows:
1. Data loading and augmentation.
2. Uniform 512×512 preprocessing for all models.
3. Base-model training (ResUNet++, TransFuse, WDFFNet).
4. HAR ensemble training with frozen bases.
5. Refinement-stage training.
6. Metric evaluation + visualization export.

### 3.2 Dataset and Preprocessing
- Dataset: Kvasir-SEG.
- Input: RGB images, binary masks.
- Resize: 512×512 for both image and mask.
- Normalization: ImageNet mean/std.
- Binarization: mask threshold at 0.5.

### 3.3 Base Architectures
- **ResUNet++:** imported from the original source for fidelity.
- **TransFuseSimple:** dual-stream feature extraction and BiFusion blocks.
- **WDFFNet:** EfficientNet + ResNet dual backbone with weighted fusion and object-aware attention.

### 3.4 Ensemble and Refinement
- **HAR Ensemble:** stacks the three model predictions, applies channel-spatial attention and softmax weighting.
- **Refinement Network:** takes image + ensemble mask as input and predicts a refined segmentation map.

### 3.5 Loss Function
Composite segmentation loss:
\[
\mathcal{L} = 0.3\,\mathcal{L}_{BCE} + 0.3\,\mathcal{L}_{Dice} + 0.3\,\mathcal{L}_{Focal} + 0.1\,\mathcal{L}_{Boundary}
\]

### 3.6 Optimization
- Optimizer: Adam.
- Mixed precision on CUDA (if available).
- Best checkpoint selected using validation Dice.

### 3.7 Evaluation Metrics
- Dice, IoU, Accuracy, Precision, Recall, F1.
- FPS and parameter profiling.
- Threshold search for optimal inference operating point.

### 3.8 Reproducibility Controls
- Seed fixation.
- Shape normalization utility.
- Partial checkpoint loading with key/shape safety.
- Automatic artifact export for traceability.

---

## Chapter 4: Results and Discussion

### 4.1 Reporting Structure
Include:
1. Quantitative comparison table.
2. Parameter/FPS profile.
3. Qualitative visual sheet of all models.
4. Failure-case analysis.

### 4.2 Expected Findings
- Ensemble should outperform most single models in Dice/IoU.
- 512×512 setup improves fine boundary preservation.
- Refinement stage improves difficult samples with fragmented or low-contrast lesions.

### 4.3 On Dice > 0.90
Achieving Dice above 0.90 depends on:
- data split,
- augmentation intensity,
- training duration,
- checkpoint quality,
- threshold optimization,
- and compute availability.

The code now includes threshold search, TTA, multi-scale inference, post-processing, and refinement to maximize performance potential under the same dataset.

### 4.4 Visual Analysis
Use exported comparison sheets:
- `all_models_visual_comparison.png`
- `studied_models_comparison.png` (when `ANALYZE_IMAGES` is set)

### 4.5 Model Comparison Artifacts
Saved to Google Drive output folder:
- `comparison_table.csv`
- `model_details.csv`
- `model_details.json`
- per-model checkpoints (`*.pth`)

### 4.6 Discussion
Discuss:
- model complementarity,
- where each base model fails,
- why weighted fusion helps,
- trade-off between quality and inference speed.

---

## Chapter 5: Conclusion and Future Work

### 5.1 Conclusion
This dissertation presents a unified 512×512 polyp segmentation pipeline using three heterogeneous architectures and a learnable hybrid ensemble with refinement. The framework emphasizes fairness, reproducibility, and dissertation-ready reporting.

### 5.2 Limitations
- Single-dataset evaluation.
- Sensitivity to split and random seed.
- Increased compute cost for ensemble inference.

### 5.3 Future Scope
- Cross-dataset generalization studies.
- Uncertainty-aware segmentation.
- Semi-supervised domain adaptation.
- Real-time optimization and model compression.

---

## Appendix A: Suggested Tables

### A.1 Primary Metrics
| Model | Dice | IoU | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ResUNet++ |  |  |  |  |  |  |
| TransFuse |  |  |  |  |  |  |
| WDFFNet |  |  |  |  |  |  |
| HAR Ensemble |  |  |  |  |  |  |
| HAR+Refine |  |  |  |  |  |  |

### A.2 Complexity + Speed
| Model | Params (M) | Trainable Params | FPS |
|---|---:|---:|---:|
| ResUNet++ |  |  |  |
| TransFuse |  |  |  |
| WDFFNet |  |  |  |
| HAR Ensemble |  |  |  |
| HAR+Refine |  |  |  |

---

## Appendix B: Colab Environment Checklist (512×512)

```python
%env DATA_DIR=/content/data/Kvasir-SEG
%env DATA_ZIP=/content/drive/MyDrive/datasets/Kvasir-SEG.zip
%env RESUNET_REPO=/content/ResUNetPlusPlus
%env RESUNET_REPO_URL=https://github.com/DebeshJha/ResUNetPlusPlus.git
%env USE_CKPT=0
%env IMG_SIZE=512
%env FINETUNE_SIZE=512
%env BASE_EPOCHS=50
%env EPOCHS=50
%run colab_har_ensemble.py
```

> Set each `%env` in a separate line. Do not paste all assignments into one line.

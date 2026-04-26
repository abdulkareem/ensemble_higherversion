# Publication Protocol (High-Impact Journal Oriented)

## Core claim
**Triple-paradigm ensemble** for colonoscopy polyp segmentation:
- VM-UNet (Mamba branch): efficient long-range sequence/state modeling
- TransFuse (Transformer branch): global contextual reasoning
- ResUNet++ (CNN branch): local edge and texture precision
- Cross-Scale Gated Fusion Module: explicit multi-scale weighting for size variation

## Required experimental stages

| Stage | Dataset(s) | Goal |
|---|---|---|
| Training | Kvasir-SEG + CVC-ClinicDB | learn robust base representations |
| Internal test | held-out 20% from training pool | primary accuracy benchmarking |
| External zero-shot | ETIS-Larib and CVC-ColonDB | prove domain generalization without retraining |

## Reporting requirements

1. Dice, IoU, Precision, Recall, Accuracy for all stages.
2. Parameter count and FPS for each branch and ensemble.
3. External zero-shot metrics are mandatory in main results table.
4. Statistical confidence intervals (bootstrap or repeated runs).

## Explainability package

1. Branchwise maps:
   - Mamba: Grad-CAM + state response proxy
   - Transformer: attention proxy map
   - CNN: Grad-CAM
2. Consensus/disagreement map from all branches.
3. Failure-case analysis with at least one low-contrast and one tiny-polyp case.

## Recommended manuscript title

**A Triple-Paradigm Ensemble for Polyp Segmentation: Leveraging Mamba Efficiency, Transformer Global Context, and CNN Boundary Precision via Cross-Scale Gated Fusion**

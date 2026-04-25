# Mamba-Fusion for Polyp Segmentation

This repository provides a research-focused implementation for a **Mamba-Fusion** segmentation ensemble:

- **VM-UNet (Mamba-style branch)**
- **TransFuse (transformer-guided branch)**
- **ResUNet++ (CNN branch)**
- **Cross-Scale Fusion Module** for scale-robust polyp prediction

## Why this setup is publishable

1. **Architecture novelty**: combines three complementary branches with an explicit cross-scale fusion head.
2. **Clinical challenge targeted**: scale variation in polyps is handled using multi-dilation adaptive fusion.
3. **Explainability (XAI)**: Grad-CAM utilities are included for qualitative justification in manuscript figures.
4. **Cross-dataset protocol**: designed to report out-of-distribution performance (e.g., train Kvasir-SEG and test ETIS / CVC-ClinicDB without retraining).

## Suggested experimental protocol

1. Train each base model on Kvasir-SEG train split.
2. Freeze the base models and train `MambaFusionEnsemble` fusion module.
3. Evaluate on:
   - Kvasir held-out test split (in-distribution)
   - CVC-ClinicDB / ETIS-Larib (cross-dataset generalization)
4. Report Dice, IoU, Precision, Recall, Accuracy.
5. Include Grad-CAM visualizations for successful and failure cases.

## Core files

- `models/vmunet_mamba.py`: VM-UNet style Mamba branch.
- `models/transfuse.py`: TransFuse branch.
- `models/resunetpp.py`: ResUNet++ wrapper branch.
- `ensemble.py`: Mamba-Fusion ensemble and Cross-Scale Fusion Module.
- `xai.py`: Grad-CAM and heatmap overlay utilities.

## Minimal usage sketch

```python
from models import VMUNetMamba, TransFuse, ResUNetPPWrapper
from ensemble import MambaFusionEnsemble

mamba = VMUNetMamba(out_size=256, pretrained=True)
transfuse = TransFuse(out_size=256, pretrained=True)
resunet = ResUNetPPWrapper("resunet++_kvasir.py", out_size=256)

model = MambaFusionEnsemble(mamba, transfuse, resunet)
```

## Reproducibility checklist for paper submission

- Fix random seed and document compute platform.
- Report number of parameters and FPS for each branch + ensemble.
- Include cross-dataset statistics with confidence intervals.
- Provide qualitative XAI maps and failure-case analysis.
- Publish code and checkpoint hashes.

# Mamba-Fusion for Polyp Segmentation

This repository now targets a **high-impact-journal-ready** workflow for colonoscopy polyp segmentation.

## 1) Proposed novelty

**Mamba-Fusion = Triple-paradigm ensemble + Cross-Scale Gated Fusion**

- **VM-UNet (Mamba)** branch: efficient sequence/state modeling.
- **TransFuse (Transformer)** branch: global contextual reasoning.
- **ResUNet++ (CNN)** branch: local edge and boundary precision.
- **Cross-Scale Gated Fusion Module (CSFM)**: explicit scale-aware gating to handle tiny, medium, and large polyps.

This "committee of experts" design is intended to combine local, global, and efficient long-context behavior in one framework.

## 2) What is implemented

- `models/vmunet_mamba.py`: VM-UNet style Mamba branch.
- `models/transfuse.py`: Transformer-guided branch.
- `models/resunetpp.py` + `resunet++_kvasir.py`: CNN branch.
- `ensemble.py`: `MambaFusionEnsemble` with `CrossScaleGatedFusionModule`.
- `xai.py`: branchwise Grad-CAM, Mamba state-response proxy, consensus/disagreement maps.
- `docs/PUBLICATION_PROTOCOL.md`: submission-grade experimental roadmap.
- `research_readiness.py`: lightweight architecture verification script.

## 3) Journal-oriented evaluation protocol

Follow `docs/PUBLICATION_PROTOCOL.md` exactly.

Minimum requirement:

1. Train on **Kvasir-SEG + CVC-ClinicDB**.
2. Validate on held-out internal splits.
3. Report **zero-shot external test** on **ETIS-Larib / CVC-ColonDB**.
4. Include branch-level explainability and failure-case analysis.

## 4) Quick start

```python
from models import VMUNetMamba, TransFuse, ResUNetPPWrapper
from ensemble import MambaFusionEnsemble

mamba = VMUNetMamba(out_size=256, pretrained=True)
transfuse = TransFuse(out_size=256, pretrained=True)
resunet = ResUNetPPWrapper("resunet++_kvasir.py", out_size=256)
model = MambaFusionEnsemble(mamba, transfuse, resunet)
```

## 5) Sanity check command

```bash
python research_readiness.py
```

> Note: this verifies architectural claims, not publication acceptance. Acceptance depends on rigorous experiments, statistical significance, and manuscript quality.


## 6) Produce publication tables from your experiment outputs

After running experiments and saving JSON metrics, build manuscript-ready tables:

```bash
python publication_results.py   --inputs outputs/metrics_kvasir.json outputs/metrics_etis.json outputs/metrics_colondb.json   --output-dir outputs/publication_bundle
```

Generated artifacts:
- `all_results_raw.csv`
- `table_main_results.csv`
- `table_external_generalization.csv`
- `table_main_results.tex` (LaTeX)
- `RESULTS_SUMMARY.md`

Use `docs/RESULTS_REPORTING_TEMPLATE.md` to draft the Results section in journal format.


## 7) Colab one-cell execution

Use `COLAB_SINGLE_CELL.md` for a single-cell Colab runner that installs dependencies, clones the repo, auto-falls back to the remote default branch if your chosen branch is missing, runs readiness checks, and optionally builds the publication result bundle.

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

Use `COLAB_SINGLE_CELL.md` for a single-cell Colab runner that mounts Google Drive, can auto-download Kvasir-SEG if needed, enforces uniform training settings for fair model comparison, and writes publication outputs + model stats to Drive with clearer error diagnostics.

## 8) Kaggle one-cell execution

Use `KAGGLE_SINGLE_CELL.md` for Kaggle-compatible execution with network-resilient behavior (best-effort pip/git and offline guidance).


## 9) Quick troubleshooting (Colab/Kaggle)

If you see:
`CalledProcessError: ... git checkout work ... returned non-zero exit status 1`
then branch `work` does not exist in the remote repo.

Fix:
- set `BRANCH = "main"`, or
- set `BRANCH` to an existing remote branch name.

The provided `COLAB_SINGLE_CELL.md` already includes fallback logic (`BRANCH` -> `origin/BRANCH` -> remote default branch).


### Journal-quality mode

Use strict quality gating when preparing submission tables:

```bash
python publication_results.py   --inputs outputs/metrics_run1.json outputs/metrics_run2.json outputs/metrics_run3.json   --output-dir outputs/publication_bundle   --min-runs-per-model 3   --min-external-datasets 1   --strict-journal-quality
```

This generates `journal_quality_check.json` and fails if minimum repeated runs or external validation requirements are not met.


### Recommended journal training depth

For high-impact submissions, run at least **50 epochs** for each base model and **50 epochs** for the ensemble head (or justify early stopping with validation curves).

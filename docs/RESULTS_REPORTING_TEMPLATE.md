# Results Reporting Template (for high-impact journal submission)

Use this template **after** generating tables with `publication_results.py`.

## 1. Main quantitative results

- Report Table 1 from `outputs/publication_bundle/table_main_results.csv`.
- Mention Dice and IoU as primary metrics, with Precision/Recall/Accuracy as supporting metrics.

Suggested sentence:

> Mamba-Fusion achieved the best overall Dice on internal benchmarks while maintaining strong IoU and balanced precision-recall behavior across datasets.

## 2. External zero-shot generalization (critical)

- Report Table 2 from `outputs/publication_bundle/table_external_generalization.csv`.
- Explicitly state **no retraining** on ETIS-Larib/CVC-ColonDB.

Suggested sentence:

> Under a strict zero-shot protocol, Mamba-Fusion ranked first in external Dice, indicating stronger robustness to cross-center domain shift.

## 3. Ablation summary

Minimum ablations to include:
1. Remove transformer branch.
2. Remove mamba branch.
3. Replace CSFM with simple average fusion.
4. Disable cross-scale gating (single-scale fusion).

## 4. Explainability package

Include branchwise overlays from `xai.py`:
- `mamba_gradcam`
- `transformer_attention_proxy`
- `cnn_gradcam`
- `consensus_map`
- `disagreement_map`

## 5. Statistical rigor

- Report mean ± std over repeated runs or confidence intervals.
- Include paired significance testing for main Dice comparison.

## 6. Failure analysis

Required examples:
- tiny/sessile polyp,
- low-contrast polyp,
- specular highlight artifact,
- motion blur frame.

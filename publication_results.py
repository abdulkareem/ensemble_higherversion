"""Build publication-ready result tables from experimental metric files.

Usage example:
python publication_results.py \
  --inputs outputs/metrics_kvasir_run1.json outputs/metrics_kvasir_run2.json outputs/metrics_etis_run1.json \
  --output-dir outputs/publication_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

PRIMARY_METRICS = ["Dice", "IoU", "Precision", "Recall", "Accuracy"]
EXTERNAL_DATASET_KEYS = ("etis", "colon", "cvc-colondb", "external")


@dataclass
class MetricsRecord:
    source_file: str
    run_id: str
    dataset: str
    model: str
    metrics: Dict[str, float]


def _guess_dataset_name(path: Path, payload: dict) -> str:
    for key in ("dataset", "dataset_name", "test_dataset"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    stem = path.stem.lower().replace("metrics", "").strip("_-")
    return stem or "unknown_dataset"


def _guess_run_id(path: Path, payload: dict) -> str:
    for key in ("run_id", "seed", "experiment_id"):
        if key in payload:
            return str(payload[key])
    return path.stem


def _extract_model_metrics(payload: dict) -> Dict[str, Dict[str, float]]:
    if "metrics_by_model" in payload and isinstance(payload["metrics_by_model"], dict):
        return payload["metrics_by_model"]
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]

    candidate = {k: v for k, v in payload.items() if isinstance(v, dict)}
    if candidate and all(any(m in v for m in PRIMARY_METRICS) for v in candidate.values()):
        return candidate

    raise ValueError("Unsupported metrics JSON format. Include metrics_by_model/results or model->metric dictionary.")


def load_records(paths: Iterable[str]) -> List[MetricsRecord]:
    records: List[MetricsRecord] = []
    for p in paths:
        path = Path(p)
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = _guess_dataset_name(path, payload)
        run_id = _guess_run_id(path, payload)
        metrics_by_model = _extract_model_metrics(payload)

        for model, metrics in metrics_by_model.items():
            safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            records.append(MetricsRecord(path.name, run_id, dataset, model, safe_metrics))
    return records


def _metric_stats(vals: List[float]) -> Dict[str, float | int | str]:
    if not vals:
        return {"mean": "", "std": "", "ci95": "", "n": 0}
    m = mean(vals)
    n = len(vals)
    s = stdev(vals) if n > 1 else 0.0
    ci95 = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "std": s, "ci95": ci95, "n": n}


def _aggregate(records: List[MetricsRecord], key_fields: List[str]) -> List[dict]:
    grouped: Dict[tuple, List[MetricsRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in key_fields)
        grouped.setdefault(key, []).append(r)

    rows = []
    for key, recs in grouped.items():
        row = {k: v for k, v in zip(key_fields, key)}
        unique_runs = sorted({r.run_id for r in recs})
        row["num_runs"] = len(unique_runs)
        for m in PRIMARY_METRICS:
            vals = [r.metrics[m] for r in recs if m in r.metrics]
            stats = _metric_stats(vals)
            row[m] = stats["mean"]
            row[f"{m}_std"] = stats["std"]
            row[f"{m}_ci95"] = stats["ci95"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[dict], header: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _format_num(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else "-"


def _latex_table(rows: List[dict]) -> str:
    newline = r"\\"
    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        "Dataset & Model & Dice (mean±std) & IoU (mean±std) & N " + newline,
        r"\hline",
    ]
    for r in rows:
        dice = f"{_format_num(r.get('Dice'))}±{_format_num(r.get('Dice_std'))}"
        iou = f"{_format_num(r.get('IoU'))}±{_format_num(r.get('IoU_std'))}"
        lines.append(f"{r.get('dataset','')} & {r.get('model','')} & {dice} & {iou} & {r.get('num_runs',0)} {newline}")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _journal_quality_check(rows: List[dict], min_runs_per_model: int, min_external_datasets: int) -> Dict[str, object]:
    datasets = sorted({r.get("dataset", "") for r in rows if r.get("dataset")})
    external = sorted({d for d in datasets if any(k in d.lower() for k in EXTERNAL_DATASET_KEYS)})

    run_failures = []
    for r in rows:
        if int(r.get("num_runs", 0)) < min_runs_per_model:
            run_failures.append(f"{r.get('dataset')}::{r.get('model')} (runs={r.get('num_runs',0)})")

    checks = {
        "min_external_datasets": {
            "required": min_external_datasets,
            "observed": len(external),
            "passed": len(external) >= min_external_datasets,
            "external_datasets": external,
        },
        "min_runs_per_model_dataset": {
            "required": min_runs_per_model,
            "passed": len(run_failures) == 0,
            "failures": run_failures,
        },
    }
    checks["overall_pass"] = checks["min_external_datasets"]["passed"] and checks["min_runs_per_model_dataset"]["passed"]
    return checks


def write_results_bundle(
    records: List[MetricsRecord],
    output_dir: str,
    min_runs_per_model: int = 3,
    min_external_datasets: int = 1,
) -> Dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for r in records:
        row = {"source_file": r.source_file, "run_id": r.run_id, "dataset": r.dataset, "model": r.model}
        for m in PRIMARY_METRICS:
            row[m] = r.metrics.get(m, "")
        raw_rows.append(row)

    main_rows = sorted(_aggregate(records, ["dataset", "model"]), key=lambda x: (x["dataset"], -(x["Dice"] or 0)))

    external_records = [r for r in records if any(k in r.dataset.lower() for k in EXTERNAL_DATASET_KEYS)]
    external_rows = _aggregate(external_records, ["model"])
    external_rows = sorted(external_rows, key=lambda x: -(x["Dice"] or 0))
    for i, row in enumerate(external_rows, start=1):
        row["rank"] = i
        row["external_dice"] = row.pop("Dice", "")
        row["external_iou"] = row.pop("IoU", "")

    header_main = ["dataset", "model", "num_runs"] + PRIMARY_METRICS + [f"{m}_std" for m in PRIMARY_METRICS] + [f"{m}_ci95" for m in PRIMARY_METRICS]
    _write_csv(out / "all_results_raw.csv", raw_rows, ["source_file", "run_id", "dataset", "model", *PRIMARY_METRICS])
    _write_csv(out / "table_main_results.csv", main_rows, header_main)
    _write_csv(out / "table_external_generalization.csv", external_rows, ["rank", "model", "num_runs", "external_dice", "external_iou", "Precision", "Recall", "Accuracy"])
    (out / "table_main_results.tex").write_text(_latex_table(main_rows), encoding="utf-8")

    quality = _journal_quality_check(main_rows, min_runs_per_model=min_runs_per_model, min_external_datasets=min_external_datasets)
    (out / "journal_quality_check.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    summary = [
        "# Results Summary",
        "",
        f"- Parsed records: {len(records)}",
        f"- Main table rows: {len(main_rows)}",
        f"- External ranking rows: {len(external_rows)}",
        f"- Journal-quality checks pass: **{quality['overall_pass']}**",
    ]
    if external_rows:
        summary.append(f"- Best external Dice model: **{external_rows[0]['model']}** ({_format_num(external_rows[0]['external_dice'])}).")
    else:
        summary.append("- No external dataset files detected by filename/dataset field.")

    (out / "RESULTS_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready result tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of metrics JSON files.")
    parser.add_argument("--output-dir", default="outputs/publication_bundle", help="Output directory.")
    parser.add_argument("--min-runs-per-model", type=int, default=3, help="Minimum repeated runs per model/dataset for quality checks.")
    parser.add_argument("--min-external-datasets", type=int, default=1, help="Minimum number of external datasets required.")
    parser.add_argument("--strict-journal-quality", action="store_true", help="Exit non-zero if journal quality checks fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    if not records:
        raise ValueError("No metric records parsed from inputs.")

    quality = write_results_bundle(
        records,
        args.output_dir,
        min_runs_per_model=args.min_runs_per_model,
        min_external_datasets=args.min_external_datasets,
    )
    print(f"[DONE] Publication bundle written to: {args.output_dir}")
    if args.strict_journal_quality and not quality["overall_pass"]:
        print("[ERROR] Journal quality checks failed. See journal_quality_check.json")
        raise SystemExit(4)


if __name__ == "__main__":
    main()

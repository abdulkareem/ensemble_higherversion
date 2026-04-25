"""Build publication-ready result tables from experimental metric files.

Usage example:
python publication_results.py \
  --inputs outputs/metrics_kvasir.json outputs/metrics_etis.json \
  --output-dir outputs/publication_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

PRIMARY_METRICS = ["Dice", "IoU", "Precision", "Recall", "Accuracy"]
EXTERNAL_DATASET_KEYS = ("etis", "colon", "cvc-colondb", "external")


@dataclass
class MetricsRecord:
    source_file: str
    dataset: str
    model: str
    metrics: Dict[str, float]


def _guess_dataset_name(path: Path, payload: dict) -> str:
    for key in ("dataset", "dataset_name", "test_dataset"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    stem = path.stem.lower().replace("metrics", "").strip("_-")
    return stem or "unknown_dataset"


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
        metrics_by_model = _extract_model_metrics(payload)

        for model, metrics in metrics_by_model.items():
            safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            records.append(MetricsRecord(path.name, dataset, model, safe_metrics))
    return records


def _aggregate(records: List[MetricsRecord], key_fields: List[str]) -> List[dict]:
    grouped: Dict[tuple, List[MetricsRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in key_fields)
        grouped.setdefault(key, []).append(r)

    rows = []
    for key, recs in grouped.items():
        row = {k: v for k, v in zip(key_fields, key)}
        for m in PRIMARY_METRICS:
            vals = [r.metrics[m] for r in recs if m in r.metrics]
            row[m] = mean(vals) if vals else ""
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[dict], header: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _format_num(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else "-"


def _latex_table(rows: List[dict]) -> str:
    newline = r"\\"
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\hline",
        "Dataset & Model & Dice & IoU & Precision & Recall & Accuracy " + newline,
        r"\hline",
    ]
    for r in rows:
        vals = [str(r.get("dataset", "")), str(r.get("model", ""))] + [_format_num(r.get(m, "")) for m in PRIMARY_METRICS]
        lines.append(" & ".join(vals) + " " + newline)
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_results_bundle(records: List[MetricsRecord], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for r in records:
        row = {"source_file": r.source_file, "dataset": r.dataset, "model": r.model}
        for m in PRIMARY_METRICS:
            row[m] = r.metrics.get(m, "")
        raw_rows.append(row)

    main_rows = sorted(_aggregate(records, ["dataset", "model"]), key=lambda x: (x["dataset"], -(x["Dice"] or 0)))

    external_records = [
        r for r in records if any(k in r.dataset.lower() for k in EXTERNAL_DATASET_KEYS)
    ]
    external_rows = _aggregate(external_records, ["model"])
    external_rows = sorted(external_rows, key=lambda x: -(x["Dice"] or 0))
    for i, row in enumerate(external_rows, start=1):
        row["rank"] = i
        row["external_dice"] = row.pop("Dice", "")
        row["external_iou"] = row.pop("IoU", "")

    _write_csv(out / "all_results_raw.csv", raw_rows, ["source_file", "dataset", "model", *PRIMARY_METRICS])
    _write_csv(out / "table_main_results.csv", main_rows, ["dataset", "model", *PRIMARY_METRICS])
    _write_csv(out / "table_external_generalization.csv", external_rows, ["rank", "model", "external_dice", "external_iou", "Precision", "Recall", "Accuracy"])
    (out / "table_main_results.tex").write_text(_latex_table(main_rows), encoding="utf-8")

    summary = [
        "# Results Summary",
        "",
        f"- Parsed records: {len(records)}",
        f"- Main table rows: {len(main_rows)}",
        f"- External ranking rows: {len(external_rows)}",
    ]
    if external_rows:
        summary.append(f"- Best external Dice model: **{external_rows[0]['model']}** ({_format_num(external_rows[0]['external_dice'])}).")
    else:
        summary.append("- No external dataset files detected by filename/dataset field.")

    (out / "RESULTS_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready result tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of metrics JSON files.")
    parser.add_argument("--output-dir", default="outputs/publication_bundle", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    if not records:
        raise ValueError("No metric records parsed from inputs.")
    write_results_bundle(records, args.output_dir)
    print(f"[DONE] Publication bundle written to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""One-command orchestrator for Colab/local quick execution.

This script focuses on repeatable checks + result packaging.
Training loops should be run separately per experiment configuration.
"""

import argparse
import subprocess
import sys
from pathlib import Path


DISCOVERY_GLOBS = ["**/*metrics*.json", "**/*results*.json"]


def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def discover_metrics(metrics_dir: str):
    base = Path(metrics_dir)
    found = []
    for pattern in DISCOVERY_GLOBS:
        found.extend(base.glob(pattern))
    # unique + deterministic order
    unique = sorted({str(p) for p in found if p.is_file()})
    return unique


def parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end sanity + publication packaging steps.")
    p.add_argument("--skip-readiness", action="store_true", help="Skip research_readiness.py")
    p.add_argument("--metrics-json", nargs="*", default=[], help="Metric JSON files for publication bundle.")
    p.add_argument("--auto-discover-metrics", action="store_true", help="Auto-discover metrics JSON files under --metrics-dir.")
    p.add_argument("--metrics-dir", default="outputs", help="Directory used for auto metric discovery.")
    p.add_argument("--require-metrics", action="store_true", help="Fail if no metric files are provided/discovered.")
    p.add_argument("--output-dir", default="outputs/publication_bundle", help="Output dir for bundled tables.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_readiness:
        run([sys.executable, "research_readiness.py"])

    metrics = list(args.metrics_json)

    if args.auto_discover_metrics:
        discovered = discover_metrics(args.metrics_dir)
        if discovered:
            print(f"[INFO] Auto-discovered {len(discovered)} metric file(s) under: {args.metrics_dir}")
            metrics.extend(discovered)

    metrics = sorted(set(metrics))

    if metrics:
        missing = [m for m in metrics if not Path(m).exists()]
        if missing:
            raise FileNotFoundError(f"Missing metrics JSON files: {missing}")

        run([
            sys.executable,
            "publication_results.py",
            "--inputs",
            *metrics,
            "--output-dir",
            args.output_dir,
        ])
    else:
        msg = (
            "No metrics JSON files found. Provide --metrics-json <files> "
            "or use --auto-discover-metrics --metrics-dir <dir>."
        )
        if args.require_metrics:
            print(f"[ERROR] {msg}")
            print("[HINT] Set --metrics-json explicitly or disable strict mode by removing --require-metrics.")
            sys.exit(2)
        print(f"[INFO] {msg}")

    print("\n[DONE] run_all.py completed.")


if __name__ == "__main__":
    main()

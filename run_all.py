"""One-command orchestrator for Colab/local quick execution.

This script intentionally focuses on repeatable checks + result packaging.
Training loops should be run separately per experiment configuration.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end sanity + publication packaging steps.")
    p.add_argument("--skip-readiness", action="store_true", help="Skip research_readiness.py")
    p.add_argument("--metrics-json", nargs="*", default=[], help="Metric JSON files for publication bundle.")
    p.add_argument("--output-dir", default="outputs/publication_bundle", help="Output dir for bundled tables.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_readiness:
        run([sys.executable, "research_readiness.py"])

    if args.metrics_json:
        missing = [m for m in args.metrics_json if not Path(m).exists()]
        if missing:
            raise FileNotFoundError(f"Missing metrics JSON files: {missing}")

        run([
            sys.executable,
            "publication_results.py",
            "--inputs",
            *args.metrics_json,
            "--output-dir",
            args.output_dir,
        ])
    else:
        print("[INFO] No --metrics-json provided; skipped publication bundle generation.")

    print("\n[DONE] run_all.py completed.")


if __name__ == "__main__":
    main()

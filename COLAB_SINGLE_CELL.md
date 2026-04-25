# Colab Single-Cell Runner

Copy-paste this **single Colab cell** to install dependencies, clone the repo, run sanity checks, and (optionally) package your experiment results for paper submission.

```python
# ======= Mamba-Fusion: Single Colab Cell (robust branch handling) =======
import os, subprocess, sys

REPO_URL = "https://github.com/abdulkareem/ensemble_higherversion.git"
REPO_DIR = "/content/ensemble_higherversion"
BRANCH = "main"   # set to your branch; if missing, code auto-falls back to remote default branch

# Optional: put your metrics json paths here after experiments
# Example: ["/content/drive/MyDrive/metrics_kvasir.json", "/content/drive/MyDrive/metrics_etis.json"]
METRICS_JSONS = []
OUTPUT_DIR = "/content/drive/MyDrive/mamba_fusion_publication_bundle"

# 1) Install runtime deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "torchaudio", "timm", "albumentations", "opencv-python", "matplotlib", "scikit-learn"
])

# 2) Clone / refresh repo
if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all", "--prune"])

# Try requested branch first, then fallback to remote default branch
checkout_ok = False
for candidate in [BRANCH, f"origin/{BRANCH}"]:
    if not candidate:
        continue
    rc = subprocess.call(["git", "-C", REPO_DIR, "checkout", candidate])
    if rc == 0:
        checkout_ok = True
        break

if not checkout_ok:
    default_branch = subprocess.check_output(
        ["git", "-C", REPO_DIR, "remote", "show", "origin"],
        text=True,
    )
    head_line = [ln for ln in default_branch.splitlines() if "HEAD branch:" in ln]
    head_branch = head_line[0].split(":", 1)[1].strip() if head_line else "main"
    subprocess.check_call(["git", "-C", REPO_DIR, "checkout", head_branch])
    print(f"[INFO] Requested branch '{BRANCH}' not found. Using default branch '{head_branch}'.")

os.chdir(REPO_DIR)

# 3) Run architecture sanity check
subprocess.check_call([sys.executable, "research_readiness.py"])

# 4) Optionally package publication tables
if METRICS_JSONS:
    cmd = [sys.executable, "run_all.py", "--skip-readiness", "--metrics-json", *METRICS_JSONS, "--output-dir", OUTPUT_DIR]
    subprocess.check_call(cmd)
    print(f"\n[DONE] Publication bundle generated at: {OUTPUT_DIR}")
else:
    print("\n[INFO] METRICS_JSONS is empty. Add your metrics files to generate publication tables.")

print("\n✅ All selected steps completed.")
# =========================================================================
```

### Notes
- If you use private repos, authenticate Git in Colab first.
- Use `publication_results.py` only with **real experimental metrics** (no fabricated values).

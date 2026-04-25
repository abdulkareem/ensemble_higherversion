# Colab Single-Cell Runner

Copy-paste this **single Colab cell** to install dependencies, clone the repo, run sanity checks, and build publication tables.

> Why your previous run finished in seconds: `METRICS_JSONS = []` means there were no result files to package, so the script correctly skipped table generation.

```python
# ======= Mamba-Fusion: Single Colab Cell (robust branch + metrics checks) =======
import os, subprocess, sys

REPO_URL = "https://github.com/abdulkareem/ensemble_higherversion.git"
REPO_DIR = "/content/ensemble_higherversion"
BRANCH = "main"   # if missing, auto-falls back to remote default branch

# If you already know metric files, list them here:
# METRICS_JSONS = ["/content/drive/MyDrive/metrics_kvasir.json", "/content/drive/MyDrive/metrics_etis.json"]
METRICS_JSONS = []

# Auto-discover metrics if METRICS_JSONS is empty
AUTO_DISCOVER_METRICS = True
METRICS_DIR = "/content/drive/MyDrive"   # where your metrics*.json/results*.json exist
REQUIRE_METRICS = True  # fail loudly instead of silently finishing when metrics are missing

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

checkout_ok = False
for candidate in [BRANCH, f"origin/{BRANCH}"]:
    rc = subprocess.call(["git", "-C", REPO_DIR, "checkout", candidate])
    if rc == 0:
        checkout_ok = True
        break

if not checkout_ok:
    info = subprocess.check_output(["git", "-C", REPO_DIR, "remote", "show", "origin"], text=True)
    head_line = [ln for ln in info.splitlines() if "HEAD branch:" in ln]
    head_branch = head_line[0].split(":", 1)[1].strip() if head_line else "main"
    subprocess.check_call(["git", "-C", REPO_DIR, "checkout", head_branch])
    print(f"[INFO] Requested branch '{BRANCH}' not found. Using default branch '{head_branch}'.")

os.chdir(REPO_DIR)

# 3) Run all checks + publication packaging
cmd = [sys.executable, "run_all.py", "--output-dir", OUTPUT_DIR]
if METRICS_JSONS:
    cmd += ["--metrics-json", *METRICS_JSONS]
if AUTO_DISCOVER_METRICS:
    cmd += ["--auto-discover-metrics", "--metrics-dir", METRICS_DIR]
if REQUIRE_METRICS:
    cmd += ["--require-metrics"]

subprocess.check_call(cmd)

print("\n✅ Completed. If no metrics are found, the cell now fails with a clear error.")
# ==============================================================================
```

### Notes
- If you use private repos, authenticate Git in Colab first.
- Use `publication_results.py` only with **real experimental metrics** (no fabricated values).
- If this fails with “No metrics JSON files found”, first run training/evaluation and save metrics files.

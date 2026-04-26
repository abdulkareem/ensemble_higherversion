# Colab Single-Cell Runner

Copy-paste this **single Colab cell** to install dependencies, clone the repo, run sanity checks, and build publication tables.

> If you get `CalledProcessError` on `git checkout work`, set `BRANCH = "main"` (or any branch that exists remotely).

> If no metrics files exist yet, this cell can run in non-strict mode and finish with a warning (no publication bundle generated).

```python
# ======= Mamba-Fusion: Single Colab Cell (robust branch + strict metrics checks) =======
import os, subprocess, sys, glob

REPO_URL = "https://github.com/abdulkareem/ensemble_higherversion.git"
REPO_DIR = "/content/ensemble_higherversion"
BRANCH = "main"   # if missing, auto-falls back to remote default branch

# If you already know metric files, list them here:
# METRICS_JSONS = ["/content/drive/MyDrive/metrics_kvasir.json", "/content/drive/MyDrive/metrics_etis.json"]
METRICS_JSONS = []

# Auto-discover metrics if METRICS_JSONS is empty
AUTO_DISCOVER_METRICS = True
METRICS_DIR = "/content/drive/MyDrive"   # where your metrics*.json/results*.json exist
ALLOW_EMPTY_METRICS = True  # True => no error when metrics are not available yet

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
if ALLOW_EMPTY_METRICS:
    cmd += ["--allow-empty-metrics"]

try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] run_all.py failed with exit code {e.returncode}.")
    found = sorted(glob.glob(os.path.join(METRICS_DIR, "**", "*metrics*.json"), recursive=True) +
                   glob.glob(os.path.join(METRICS_DIR, "**", "*results*.json"), recursive=True))
    print(f"[DEBUG] Found {len(found)} metric-like files under {METRICS_DIR}.")
    if found:
        print("[DEBUG] Sample files:")
        for p in found[:10]:
            print(" -", p)
    else:
        print("[HINT] No metrics files found yet. Run training/evaluation first, then re-run this cell.")
    raise

bundle_file = os.path.join(OUTPUT_DIR, "table_main_results.csv")
if os.path.exists(bundle_file):
    print(f"\n✅ Completed with publication bundle: {bundle_file}")
else:
    print("\n⚠️ Completed sanity run only. No metrics JSON found, so no publication bundle was generated.")
# ==============================================================================
```

### Notes
- If you use private repos, authenticate Git in Colab first.
- Use `publication_results.py` only with **real experimental metrics** (no fabricated values).
- Set `ALLOW_EMPTY_METRICS = True` only if you intentionally want a quick non-failing smoke run.

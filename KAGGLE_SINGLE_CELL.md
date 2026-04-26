# Kaggle Single-Cell Runner (Network-Resilient)

Kaggle notebooks often run with internet restrictions. The warnings you saw (`Temporary failure in name resolution`) mean the environment could not reach PyPI/GitHub at that moment.

Use this **Kaggle-safe single cell**:

```python
import os, sys, glob, subprocess

IS_KAGGLE = os.path.exists('/kaggle')
REPO_URL = "https://github.com/abdulkareem/ensemble_higherversion.git"
REPO_DIR = "/kaggle/working/ensemble_higherversion" if IS_KAGGLE else "/content/ensemble_higherversion"
BRANCH = "main"

METRICS_JSONS = []
AUTO_DISCOVER_METRICS = True
METRICS_DIR = "/kaggle/working" if IS_KAGGLE else "/content"
ALLOW_EMPTY_METRICS = False
OUTPUT_DIR = os.path.join(METRICS_DIR, "mamba_fusion_publication_bundle")


def run(cmd, allow_fail=False):
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        print(f"[WARN] Command failed: {' '.join(cmd)}")
        print(f"[WARN] {e}")
        if not allow_fail:
            raise
        return False

# 1) Dependency step (best-effort in offline Kaggle)
run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], allow_fail=True)
run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "torchaudio", "timm", "albumentations", "opencv-python", "matplotlib", "scikit-learn"
], allow_fail=True)

# 2) Clone/update repo (if internet blocked, use existing local copy)
if not os.path.exists(REPO_DIR):
    cloned = run(["git", "clone", REPO_URL, REPO_DIR], allow_fail=True)
    if not cloned:
        raise RuntimeError(
            "Could not clone repository (likely no internet). "
            "Upload the repo as a Kaggle Dataset or enable internet in notebook settings."
        )

run(["git", "-C", REPO_DIR, "fetch", "--all", "--prune"], allow_fail=True)

checkout_ok = run(["git", "-C", REPO_DIR, "checkout", BRANCH], allow_fail=True)
if not checkout_ok:
    checkout_ok = run(["git", "-C", REPO_DIR, "checkout", f"origin/{BRANCH}"], allow_fail=True)

if not checkout_ok:
    info = subprocess.check_output(["git", "-C", REPO_DIR, "remote", "show", "origin"], text=True)
    head_line = [ln for ln in info.splitlines() if "HEAD branch:" in ln]
    head_branch = head_line[0].split(":", 1)[1].strip() if head_line else "main"
    run(["git", "-C", REPO_DIR, "checkout", head_branch])
    print(f"[INFO] Requested branch '{BRANCH}' not found. Using '{head_branch}'.")

os.chdir(REPO_DIR)

# 3) Run checks + publication packaging
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
    for p in found[:10]:
        print(" -", p)
    if not found:
        print("[HINT] No metrics files yet. Run training/evaluation first.")
    raise

print(f"\n✅ Completed. Publication bundle at: {OUTPUT_DIR}")
```

## If internet is OFF in Kaggle
1. Turn internet ON in notebook settings **or**
2. Upload this repo as a Kaggle Dataset and set `REPO_DIR` to that mounted path.

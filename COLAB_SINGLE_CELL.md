# Colab Single-Cell Runner

Copy-paste this **single Colab cell** to install dependencies, clone the repo, run sanity checks, and (optionally) package your experiment results for paper submission.

```python
# ======= Mamba-Fusion: Single Colab Cell =======
import os, subprocess, sys, textwrap

REPO_URL = "https://github.com/<your-user>/<your-repo>.git"   # <-- change this
REPO_DIR = "/content/ensemble_higherversion"
BRANCH = "work"  # or your target branch

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

subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all"])
subprocess.check_call(["git", "-C", REPO_DIR, "checkout", BRANCH])

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
# ===============================================
```

### Notes
- If you use private repos, authenticate Git in Colab first.
- Use `publication_results.py` only with **real experimental metrics** (no fabricated values).

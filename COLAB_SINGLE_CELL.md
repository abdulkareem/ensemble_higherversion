# Colab Single-Cell Runner (Train All + Publishable Outputs)

This cell will:
1. mount Google Drive,
2. install deps,
3. clone/update repo,
4. auto-download Kvasir-SEG (optional),
5. train **VMUNetMamba**, **TransFuse**, **ResUNet++** with uniform training settings,
6. train **Mamba-Fusion ensemble head**,
7. generate publication tables,
8. save everything in Google Drive.

```python
import os, subprocess, sys

# Optional: mount Drive if not mounted already
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print('[WARN] Could not mount Google Drive automatically:', e)

REPO_URL = "https://github.com/abdulkareem/ensemble_higherversion.git"
REPO_DIR = "/content/ensemble_higherversion"
BRANCH = "main"

# ===== REQUIRED / DATA =====
KVASIR_DIR = "/content/drive/MyDrive/Kvasir-SEG"  # preferred persistent location
CVC_DIR = ""  # e.g. "/content/drive/MyDrive/CVC-ClinicDB"
TRAIN_DATA_DIRS = [KVASIR_DIR] + ([CVC_DIR] if CVC_DIR else [])

AUTO_DOWNLOAD_KVASIR = True
KVASIR_ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_FALLBACK_DIR = "/content/data/Kvasir-SEG"

# Optional external dataset for zero-shot testing (same folder structure)
EXTERNAL_DATA_DIR = ""  # e.g. "/content/drive/MyDrive/ETIS-Larib"

OUTPUT_DIR = "/content/drive/MyDrive/mamba_fusion_publication_bundle"

# ===== Uniform training settings for fair comparison =====
BASE_EPOCHS = 50
ENSEMBLE_EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SIZE = 256
NUM_WORKERS = 2
SEED = 42
LR = 1e-4

# 1) Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "torchaudio", "timm", "albumentations", "opencv-python", "matplotlib", "scikit-learn", "pandas", "tabulate", "gdown"
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
    print(f"[INFO] Requested branch '{BRANCH}' not found. Using default '{head_branch}'.")

os.chdir(REPO_DIR)

# 3) Auto-download Kvasir-SEG if KVASIR missing
if (not os.path.isdir(KVASIR_DIR)) and AUTO_DOWNLOAD_KVASIR:
    os.makedirs('/content/data', exist_ok=True)
    zip_path = '/content/data/kvasir-seg.zip'
    subprocess.check_call(['wget', '--no-check-certificate', KVASIR_ZIP_URL, '-O', zip_path])
    subprocess.check_call(['unzip', '-qq', '-o', zip_path, '-d', '/content/data'])
    if os.path.isdir(KVASIR_FALLBACK_DIR):
        TRAIN_DATA_DIRS = [KVASIR_FALLBACK_DIR] + ([CVC_DIR] if CVC_DIR else [])
        print(f"[INFO] KVASIR dir switched to downloaded dataset: {KVASIR_FALLBACK_DIR}")

# 4) Preflight checks
if not os.path.exists("train_all.py"):
    raise FileNotFoundError("train_all.py not found. Check REPO_DIR/branch checkout.")
for d in TRAIN_DATA_DIRS:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Training dataset dir not found: {d}")
    if not os.path.isdir(os.path.join(d, "images")) or not os.path.isdir(os.path.join(d, "masks")):
        raise FileNotFoundError(f"Dataset {d} must contain 'images/' and 'masks/' folders.")

# 5) Train all models + ensemble (uniform settings)
cmd = [
    sys.executable, "train_all.py",
    "--train-data-dirs", *TRAIN_DATA_DIRS,
    "--output-dir", OUTPUT_DIR,
    "--base-epochs", str(BASE_EPOCHS),
    "--ensemble-epochs", str(ENSEMBLE_EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--image-size", str(IMAGE_SIZE),
    "--num-workers", str(NUM_WORKERS),
    "--seed", str(SEED),
    "--lr", str(LR),
]
if EXTERNAL_DATA_DIR:
    cmd += ["--external-data-dir", EXTERNAL_DATA_DIR]

proc = subprocess.run(cmd, text=True, capture_output=True)
if proc.returncode != 0:
    print("\n[ERROR] train_all.py failed.")
    print("[STDOUT]\n", proc.stdout[-4000:])
    print("[STDERR]\n", proc.stderr[-4000:])
    raise RuntimeError(f"train_all.py failed with exit code {proc.returncode}")

# 6) Build publishable tables
metric_files = [os.path.join(OUTPUT_DIR, "metrics_internal.json")]
ext_file = os.path.join(OUTPUT_DIR, "metrics_external.json")
if os.path.exists(ext_file):
    metric_files.append(ext_file)

cmd_pub = [
    sys.executable, "publication_results.py",
    "--inputs", *metric_files,
    "--output-dir", OUTPUT_DIR,
    "--min-runs-per-model", "1",
    "--min-external-datasets", "1",
]
subprocess.check_call(cmd_pub)

print("\n✅ Training + publication packaging completed.")
print("Outputs saved to:", OUTPUT_DIR)
print("Model stats saved to:", os.path.join(OUTPUT_DIR, "model_stats.json"))
```

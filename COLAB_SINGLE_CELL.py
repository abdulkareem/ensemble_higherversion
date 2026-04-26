"""Colab one-file runner (no markdown fences) to avoid notebook syntax issues.

Usage in a Colab cell:
!python COLAB_SINGLE_CELL.py

Set environment variables to override defaults, e.g.:
%env KVASIR_DIR=/content/drive/MyDrive/Kvasir-SEG
%env CVC_DIR=/content/drive/MyDrive/CVC-ClinicDB
%env ETIS_DIR=/content/drive/MyDrive/ETIS-Larib
%env COLONDB_DIR=/content/drive/MyDrive/CVC-ColonDB
"""

import glob
import os
import subprocess
import sys


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


try:
    from google.colab import drive

    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print('[WARN] Could not mount Google Drive automatically:', e)

REPO_URL = env("REPO_URL", "https://github.com/abdulkareem/ensemble_higherversion.git")
REPO_DIR = env("REPO_DIR", "/content/ensemble_higherversion")
BRANCH = env("BRANCH", "main")

KVASIR_DIR = env("KVASIR_DIR", "/content/drive/MyDrive/Kvasir-SEG")
CVC_DIR = env("CVC_DIR", "")
TRAIN_DATA_DIRS = [d for d in [KVASIR_DIR, CVC_DIR] if d]

AUTO_DOWNLOAD_KVASIR = env("AUTO_DOWNLOAD_KVASIR", "true").lower() == "true"
KVASIR_ZIP_URL = env("KVASIR_ZIP_URL", "https://datasets.simula.no/downloads/kvasir-seg.zip")
KVASIR_FALLBACK_DIR = env("KVASIR_FALLBACK_DIR", "/content/data/Kvasir-SEG")

ETIS_DIR = env("ETIS_DIR", "")
COLONDB_DIR = env("COLONDB_DIR", "")
EXTERNAL_DATA_DIRS = [d for d in [ETIS_DIR, COLONDB_DIR] if d]

OUTPUT_DIR = env("OUTPUT_DIR", "/content/drive/MyDrive/mamba_fusion_publication_bundle")
BASE_EPOCHS = env("BASE_EPOCHS", "50")
ENSEMBLE_EPOCHS = env("ENSEMBLE_EPOCHS", "50")
BATCH_SIZE = env("BATCH_SIZE", "8")
IMAGE_SIZE = env("IMAGE_SIZE", "256")
NUM_WORKERS = env("NUM_WORKERS", "2")
SEED = env("SEED", "42")
LR = env("LR", "1e-4")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "albumentations",
    "opencv-python",
    "matplotlib",
    "scikit-learn",
    "pandas",
    "tabulate",
    "gdown",
])

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

if (not os.path.isdir(KVASIR_DIR)) and AUTO_DOWNLOAD_KVASIR:
    os.makedirs('/content/data', exist_ok=True)
    zip_path = '/content/data/kvasir-seg.zip'
    subprocess.check_call(['wget', '--no-check-certificate', KVASIR_ZIP_URL, '-O', zip_path])
    subprocess.check_call(['unzip', '-qq', '-o', zip_path, '-d', '/content/data'])
    if os.path.isdir(KVASIR_FALLBACK_DIR):
        TRAIN_DATA_DIRS = [KVASIR_FALLBACK_DIR] + ([CVC_DIR] if CVC_DIR else [])
        print(f"[INFO] KVASIR dir switched to downloaded dataset: {KVASIR_FALLBACK_DIR}")

if not os.path.exists("train_all.py"):
    raise FileNotFoundError("train_all.py not found. Check REPO_DIR/branch checkout.")
for d in TRAIN_DATA_DIRS:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Training dataset dir not found: {d}")
    if not os.path.isdir(os.path.join(d, "images")) or not os.path.isdir(os.path.join(d, "masks")):
        raise FileNotFoundError(f"Dataset {d} must contain 'images/' and 'masks/' folders.")

help_text = subprocess.check_output([sys.executable, "train_all.py", "--help"], text=True)
if "--train-data-dirs" in help_text:
    cmd = [
        sys.executable,
        "train_all.py",
        "--train-data-dirs",
        *TRAIN_DATA_DIRS,
        "--output-dir",
        OUTPUT_DIR,
        "--base-epochs",
        BASE_EPOCHS,
        "--ensemble-epochs",
        ENSEMBLE_EPOCHS,
        "--batch-size",
        BATCH_SIZE,
        "--image-size",
        IMAGE_SIZE,
        "--num-workers",
        NUM_WORKERS,
        "--seed",
        SEED,
        "--lr",
        LR,
    ]
else:
    cmd = [
        sys.executable,
        "train_all.py",
        "--data-dir",
        TRAIN_DATA_DIRS[0],
        "--output-dir",
        OUTPUT_DIR,
        "--base-epochs",
        BASE_EPOCHS,
        "--ensemble-epochs",
        ENSEMBLE_EPOCHS,
        "--batch-size",
        BATCH_SIZE,
        "--image-size",
        IMAGE_SIZE,
        "--num-workers",
        NUM_WORKERS,
        "--seed",
        SEED,
        "--lr",
        LR,
    ]

if EXTERNAL_DATA_DIRS:
    cmd += ["--external-data-dirs", *EXTERNAL_DATA_DIRS]

proc = subprocess.run(cmd, text=True, capture_output=True)
if proc.returncode != 0:
    print("\n[ERROR] train_all.py failed.")
    print("[STDOUT]\n", proc.stdout[-4000:])
    print("[STDERR]\n", proc.stderr[-4000:])
    raise RuntimeError(f"train_all.py failed with exit code {proc.returncode}")

metric_files = [os.path.join(OUTPUT_DIR, "metrics_internal.json")]
metric_files += sorted(glob.glob(os.path.join(OUTPUT_DIR, "metrics_external_*.json")))

cmd_pub = [
    sys.executable,
    "publication_results.py",
    "--inputs",
    *metric_files,
    "--output-dir",
    OUTPUT_DIR,
    "--min-runs-per-model",
    "1",
    "--min-external-datasets",
    "2",
]
subprocess.check_call(cmd_pub)

print("\n✅ Training + publication packaging completed.")
print("Outputs saved to:", OUTPUT_DIR)
print("Model stats saved to:", os.path.join(OUTPUT_DIR, "model_stats.json"))

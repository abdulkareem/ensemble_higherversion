"""Generate publication figures: metric charts + qualitative image comparisons."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import KvasirSegDataset
from ensemble import MambaFusionEnsemble
from models import ResUNetPPWrapper, TransFuse, VMUNetMamba
from utils import ensure_binary_output


def parse_args():
    p = argparse.ArgumentParser(description="Generate qualitative and quantitative publication figures.")
    p.add_argument("--metrics-json", required=True, help="Path to metrics_internal.json")
    p.add_argument("--data-dir", required=True, help="Dataset root with images/ and masks/")
    p.add_argument("--output-dir", required=True, help="Output directory for figures")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _collect_pairs(data_dir: str) -> pd.DataFrame:
    images = sorted(glob(str(Path(data_dir) / "images" / "*.jpg")))
    if not images:
        images = sorted(glob(str(Path(data_dir) / "images" / "*.png")))
    masks = sorted(glob(str(Path(data_dir) / "masks" / "*.jpg")))
    if not masks:
        masks = sorted(glob(str(Path(data_dir) / "masks" / "*.png")))
    if len(images) != len(masks) or len(images) == 0:
        raise ValueError(f"Invalid dataset at {data_dir}; expected matching images/masks")
    return pd.DataFrame({"image": images, "mask": masks})


def _load_models(payload: dict, device: torch.device, image_size: int) -> Dict[str, torch.nn.Module]:
    ckpt = payload.get("checkpoints", {})

    mamba = VMUNetMamba(out_size=image_size, pretrained=False).to(device)
    transfuse = TransFuse(out_size=image_size, pretrained=False).to(device)
    resunet = ResUNetPPWrapper("resunet++_kvasir.py", out_size=image_size).to(device)

    if "VMUNetMamba" in ckpt:
        mamba.load_state_dict(torch.load(ckpt["VMUNetMamba"], map_location=device), strict=False)
    if "TransFuse" in ckpt:
        transfuse.load_state_dict(torch.load(ckpt["TransFuse"], map_location=device), strict=False)
    if "ResUNet++" in ckpt:
        resunet.load_state_dict(torch.load(ckpt["ResUNet++"], map_location=device), strict=False)

    ensemble = MambaFusionEnsemble(mamba, transfuse, resunet).to(device)
    if "Mamba-Fusion" in ckpt:
        ensemble.load_state_dict(torch.load(ckpt["Mamba-Fusion"], map_location=device), strict=False)

    models = {
        "VMUNetMamba": mamba.eval(),
        "TransFuse": transfuse.eval(),
        "ResUNet++": resunet.eval(),
        "Mamba-Fusion": ensemble.eval(),
    }
    return models


def _denorm(img: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(img * std + mean, 0, 1)


def _save_metric_bar(metrics_by_model: Dict[str, Dict[str, float]], out_dir: Path):
    names = list(metrics_by_model.keys())
    dice = [metrics_by_model[n].get("Dice", 0.0) for n in names]
    iou = [metrics_by_model[n].get("IoU", 0.0) for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, dice, width=w, label="Dice")
    ax.bar(x + w / 2, iou, width=w, label="IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Internal Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar_dice_iou.png", dpi=300)
    plt.close(fig)


def _save_qualitative_grid(models: Dict[str, torch.nn.Module], loader: DataLoader, device: torch.device, out_dir: Path, num_samples: int):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    rows = min(num_samples, x.shape[0])

    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            logits = ensure_binary_output(model(x), size=y.shape[-2:])
            preds[name] = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    x_np = x.cpu().permute(0, 2, 3, 1).numpy()
    y_np = y.cpu().numpy()

    cols = 2 + len(models)  # input + gt + model preds
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    model_names = list(models.keys())
    for i in range(rows):
        axes[i, 0].imshow(_denorm(x_np[i]))
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(y_np[i, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth")

        for j, mname in enumerate(model_names, start=2):
            axes[i, j].imshow(preds[mname][i, 0], cmap="gray")
            axes[i, j].set_title(mname)

        for c in range(cols):
            axes[i, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_comparison_grid.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    metrics_by_model = payload.get("metrics_by_model", {})
    if not metrics_by_model:
        raise ValueError("metrics_json must contain metrics_by_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = _load_models(payload, device=device, image_size=args.image_size)

    df = _collect_pairs(args.data_dir)
    ds = KvasirSegDataset(df, image_size=args.image_size, augment=False)
    loader = DataLoader(ds, batch_size=max(args.num_samples, 2), shuffle=True)

    _save_metric_bar(metrics_by_model, out_dir)
    _save_qualitative_grid(models, loader, device, out_dir, num_samples=args.num_samples)

    summary = {
        "figures": [
            str(out_dir / "metrics_bar_dice_iou.png"),
            str(out_dir / "qualitative_comparison_grid.png"),
        ]
    }
    (out_dir / "figures_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

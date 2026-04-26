"""End-to-end training script for Mamba-Fusion publication workflow.

Trains base models (VMUNetMamba, TransFuse, ResUNet++) and then trains the
Cross-Scale fusion head. Saves checkpoints + metrics JSON for publication bundling.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DataConfig, KvasirSegDataset, make_dataloaders
from ensemble import MambaFusionEnsemble, train_ensemble_head
from evaluate import evaluate_model
from models import ResUNetPPWrapper, TransFuse, VMUNetMamba
from train import train_model
from utils import seed_everything


def _evaluate_all(models: Dict[str, torch.nn.Module], loader, device: torch.device) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, model in models.items():
        out[name] = evaluate_model(model.to(device), loader, device)
    return out


def _external_loader(data_dir: str, image_size: int, batch_size: int, num_workers: int):
    import os
    from glob import glob

    images = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    if not images:
        images = sorted(glob(os.path.join(data_dir, "images", "*.png")))
    masks = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))
    if not masks:
        masks = sorted(glob(os.path.join(data_dir, "masks", "*.png")))
    if len(images) != len(masks) or len(images) == 0:
        raise ValueError(f"Invalid external dataset at {data_dir}; expected matching images/masks.")

    df = pd.DataFrame({"image": images, "mask": masks})
    ds = KvasirSegDataset(df, image_size=image_size, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def parse_args():
    p = argparse.ArgumentParser(description="Train all models + ensemble and export publication-ready metrics.")
    p.add_argument("--data-dir", required=True, help="Primary dataset root (expects images/ and masks/).")
    p.add_argument("--external-data-dir", default="", help="Optional external dataset for zero-shot evaluation.")
    p.add_argument("--output-dir", default="outputs/full_run", help="Directory for checkpoints and metric files.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--base-epochs", type=int, default=50)
    p.add_argument("--ensemble-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained encoders.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    pretrained = not args.no_pretrained

    mamba = VMUNetMamba(out_size=args.image_size, pretrained=pretrained)
    transfuse = TransFuse(out_size=args.image_size, pretrained=pretrained)
    resunet = ResUNetPPWrapper("resunet++_kvasir.py", out_size=args.image_size)

    print("[INFO] Training VMUNetMamba...")
    _, mamba_ckpt = train_model(
        mamba,
        train_loader,
        val_loader,
        device,
        epochs=args.base_epochs,
        lr=args.lr,
        save_name=str(out / "best_vmunet_mamba.pth"),
    )

    print("[INFO] Training TransFuse...")
    _, transfuse_ckpt = train_model(
        transfuse,
        train_loader,
        val_loader,
        device,
        epochs=args.base_epochs,
        lr=args.lr,
        save_name=str(out / "best_transfuse.pth"),
    )

    print("[INFO] Training ResUNet++...")
    _, resunet_ckpt = train_model(
        resunet,
        train_loader,
        val_loader,
        device,
        epochs=args.base_epochs,
        lr=args.lr,
        save_name=str(out / "best_resunetpp.pth"),
    )

    ensemble = MambaFusionEnsemble(mamba, transfuse, resunet)
    print("[INFO] Training Mamba-Fusion head...")
    _, ensemble_ckpt = train_ensemble_head(
        ensemble,
        train_loader,
        val_loader,
        device,
        epochs=args.ensemble_epochs,
        lr=args.lr,
    )
    # copy ensemble ckpt into output dir for consistency
    shutil.copy2(ensemble_ckpt, out / "best_mamba_fusion_ensemble.pth")

    models = {
        "VMUNetMamba": mamba,
        "TransFuse": transfuse,
        "ResUNet++": resunet,
        "Mamba-Fusion": ensemble,
    }

    print("[INFO] Evaluating on internal test split...")
    internal_metrics = _evaluate_all(models, test_loader, device)
    internal_payload = {
        "dataset": "internal_test",
        "run_id": f"seed_{args.seed}",
        "metrics_by_model": internal_metrics,
        "checkpoints": {
            "VMUNetMamba": mamba_ckpt,
            "TransFuse": transfuse_ckpt,
            "ResUNet++": resunet_ckpt,
            "Mamba-Fusion": str(out / "best_mamba_fusion_ensemble.pth"),
        },
    }
    (out / "metrics_internal.json").write_text(json.dumps(internal_payload, indent=2), encoding="utf-8")

    if args.external_data_dir:
        print(f"[INFO] Evaluating external dataset: {args.external_data_dir}")
        ext_loader = _external_loader(args.external_data_dir, args.image_size, args.batch_size, args.num_workers)
        external_metrics = _evaluate_all(models, ext_loader, device)
        external_payload = {
            "dataset": Path(args.external_data_dir).name,
            "run_id": f"seed_{args.seed}",
            "metrics_by_model": external_metrics,
        }
        (out / "metrics_external.json").write_text(json.dumps(external_payload, indent=2), encoding="utf-8")

    print(f"[DONE] Training and evaluation complete. Outputs in: {out}")


if __name__ == "__main__":
    main()

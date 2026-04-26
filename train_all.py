"""End-to-end training script for Mamba-Fusion publication workflow.

Trains base models (VMUNetMamba, TransFuse, ResUNet++) and then trains the
Cross-Scale fusion head. Saves checkpoints + metrics JSON for publication bundling.
"""

from __future__ import annotations

import argparse
import json
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import KvasirSegDataset
from ensemble import MambaFusionEnsemble, train_ensemble_head
from evaluate import evaluate_model
from models import ResUNetPPWrapper, TransFuse, VMUNetMamba
from train import train_model
from utils import parameter_stats, seed_everything


def _model_stats(models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, int]]:
    stats = {}
    for name, model in models.items():
        total, trainable = parameter_stats(model)
        stats[name] = {"total_params": int(total), "trainable_params": int(trainable)}
    return stats


def _evaluate_all(models: Dict[str, torch.nn.Module], loader, device: torch.device) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, model in models.items():
        out[name] = evaluate_model(model.to(device), loader, device)
    return out


def _collect_pairs(data_dir: str) -> pd.DataFrame:
    images = sorted(glob(str(Path(data_dir) / "images" / "*.jpg")))
    if not images:
        images = sorted(glob(str(Path(data_dir) / "images" / "*.png")))
    masks = sorted(glob(str(Path(data_dir) / "masks" / "*.jpg")))
    if not masks:
        masks = sorted(glob(str(Path(data_dir) / "masks" / "*.png")))

    if len(images) != len(masks) or len(images) == 0:
        raise ValueError(f"Invalid dataset at {data_dir}; expected matching images/masks.")

    return pd.DataFrame({"image": images, "mask": masks, "source_dataset": Path(data_dir).name})


def _build_combined_loaders(
    data_dirs: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
):
    frames = [_collect_pairs(d) for d in data_dirs]
    df = pd.concat(frames, ignore_index=True)

    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio / (1 - test_ratio), random_state=seed, shuffle=True)

    train_ds = KvasirSegDataset(train_df.reset_index(drop=True), image_size=image_size, augment=True)
    val_ds = KvasirSegDataset(val_df.reset_index(drop=True), image_size=image_size, augment=False)
    test_ds = KvasirSegDataset(test_df.reset_index(drop=True), image_size=image_size, augment=False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, sorted({Path(d).name for d in data_dirs})


def _external_loader(data_dir: str, image_size: int, batch_size: int, num_workers: int):
    df = _collect_pairs(data_dir)
    ds = KvasirSegDataset(df, image_size=image_size, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def parse_args():
    p = argparse.ArgumentParser(description="Train all models + ensemble and export publication-ready metrics.")
    p.add_argument("--data-dir", default="", help="Single primary dataset root (images/ and masks/).")
    p.add_argument("--train-data-dirs", nargs="*", default=[], help="Multiple dataset roots for joint training (recommended).")
    p.add_argument("--external-data-dir", default="", help="Optional external dataset for zero-shot evaluation.")
    p.add_argument("--output-dir", default="outputs/full_run", help="Directory for checkpoints and metric files.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--base-epochs", type=int, default=50)
    p.add_argument("--ensemble-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained encoders.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dirs = args.train_data_dirs if args.train_data_dirs else ([args.data_dir] if args.data_dir else [])
    if not train_dirs:
        raise ValueError("Provide --data-dir or --train-data-dirs for training.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, source_datasets = _build_combined_loaders(
        train_dirs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    pretrained = not args.no_pretrained

    mamba = VMUNetMamba(out_size=args.image_size, pretrained=pretrained)
    transfuse = TransFuse(out_size=args.image_size, pretrained=pretrained)
    resunet = ResUNetPPWrapper("resunet++_kvasir.py", out_size=args.image_size)

    print("[INFO] Training VMUNetMamba...")
    _, mamba_ckpt = train_model(mamba, train_loader, val_loader, device, epochs=args.base_epochs, lr=args.lr, save_name=str(out / "best_vmunet_mamba.pth"))

    print("[INFO] Training TransFuse...")
    _, transfuse_ckpt = train_model(transfuse, train_loader, val_loader, device, epochs=args.base_epochs, lr=args.lr, save_name=str(out / "best_transfuse.pth"))

    print("[INFO] Training ResUNet++...")
    _, resunet_ckpt = train_model(resunet, train_loader, val_loader, device, epochs=args.base_epochs, lr=args.lr, save_name=str(out / "best_resunetpp.pth"))

    ensemble = MambaFusionEnsemble(mamba, transfuse, resunet)
    print("[INFO] Training Mamba-Fusion head...")
    _, ensemble_ckpt = train_ensemble_head(ensemble, train_loader, val_loader, device, epochs=args.ensemble_epochs, lr=args.lr)
    shutil.copy2(ensemble_ckpt, out / "best_mamba_fusion_ensemble.pth")

    models = {"VMUNetMamba": mamba, "TransFuse": transfuse, "ResUNet++": resunet, "Mamba-Fusion": ensemble}

    uniform_training_config = {
        "source_datasets": source_datasets,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "base_epochs": args.base_epochs,
        "ensemble_epochs": args.ensemble_epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    model_stats = _model_stats(models)
    (out / "model_stats.json").write_text(json.dumps(model_stats, indent=2), encoding="utf-8")

    print("[INFO] Evaluating on internal test split...")
    internal_metrics = _evaluate_all(models, test_loader, device)
    internal_payload = {
        "dataset": "internal_test",
        "run_id": f"seed_{args.seed}",
        "metrics_by_model": internal_metrics,
        "uniform_training_config": uniform_training_config,
        "model_stats": model_stats,
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
            "uniform_training_config": uniform_training_config,
        }
        (out / "metrics_external.json").write_text(json.dumps(external_payload, indent=2), encoding="utf-8")

    print(f"[DONE] Training and evaluation complete. Outputs in: {out}")


if __name__ == "__main__":
    main()

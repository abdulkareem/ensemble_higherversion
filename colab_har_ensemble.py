"""
Colab-ready Hybrid Attention Ensemble (HAR) pipeline for Kvasir-SEG.

This script is intentionally organized into notebook-style cells so you can paste
it into Google Colab cell-by-cell.
"""

# =========================
# Cell 1: Setup
# =========================
# !pip install -q timm albumentations pandas tabulate gdown

import os
import random
import subprocess
import importlib.util
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from glob import glob
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

import albumentations as A
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cuda_is_usable() -> bool:
    """
    Some Colab sessions report CUDA as available but have an invalid CUDA context.
    Run a tiny allocation + op to verify the device is healthy before using it.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        _ = x + 1
        return True
    except Exception as e:
        print(f"[WARN] CUDA reported available but failed health check: {type(e).__name__}: {e}")
        return False


def seed_everything(seed: int = 42, use_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        # torch.manual_seed may touch CUDA generators in some environments.
        torch.manual_seed(seed)
    except Exception as e:
        print(f"[WARN] torch.manual_seed failed, seeding CPU default generator only: {type(e).__name__}: {e}")
        torch.random.default_generator.manual_seed(seed)
    if use_cuda:
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception as e:
            # Keeps the script runnable on sessions where CUDA initialization is unstable.
            print(f"[WARN] CUDA seeding failed, continuing without cuda.manual_seed_all: {type(e).__name__}: {e}")
    # deterministic=True can trigger "unable to find an engine" errors for some ops.
    # Keep training/evaluation stable via seeding while allowing cuDNN to select valid kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
CUDA_OK = _cuda_is_usable() and not FORCE_CPU
seed_everything(42, use_cuda=CUDA_OK)
if CUDA_OK:
    try:
        # One more guard: this catches sessions with a poisoned CUDA context.
        _ = torch.tensor([1.0], device="cuda") * 2.0
        DEVICE = torch.device("cuda")
    except Exception as e:
        print(f"[WARN] CUDA warmup failed, falling back to CPU: {type(e).__name__}: {e}")
        CUDA_OK = False
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")
USE_AMP = DEVICE.type == "cuda" and os.environ.get("DISABLE_AMP", "0") != "1"
print(f"Device: {DEVICE} | AMP: {USE_AMP}")
OUTPUT_DIR = Path("/content/drive/MyDrive/ensemble_outputs")


# =========================
# Cell 2: Repo clone (Colab)
# =========================

def clone_repo_if_needed(repo_url: str, target_dir: str) -> None:
    if not os.path.exists(target_dir):
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    else:
        print(f"Repo already exists: {target_dir}")


# Example for Colab:
# REPO_DIR = "/content/ensembleArchitectureBalkees"
# clone_repo_if_needed("https://github.com/abdulkareem/ensembleArchitectureBalkees.git", REPO_DIR)


# =========================
# Cell 3: Exact model architectures (clean Python classes)
# =========================

class ObjectAwareAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, coarse: Optional[torch.Tensor] = None) -> torch.Tensor:
        ch = self.channel_att(feat) * feat
        if coarse is None:
            max_pool = torch.max(feat, dim=1, keepdim=True)[0]
            avg_pool = torch.mean(feat, dim=1, keepdim=True)
            sp = self.spatial_conv(torch.cat([max_pool, avg_pool], dim=1))
        else:
            sp = F.interpolate(coarse, size=feat.shape[2:], mode="bilinear", align_corners=False)
        return feat * ch * sp


class WeightedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_b = nn.Conv2d(channels, channels, kernel_size=1)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_proj = self.conv_a(a)
        b_proj = self.conv_b(b)
        w = self.weight_conv(torch.cat([a_proj, b_proj], dim=1))
        return a_proj * w + b_proj * (1.0 - w)


class WDFFNet(nn.Module):
    """Architecture kept aligned with training notebook implementation."""

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.back_a = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        self.back_b = timm.create_model(
            "resnet50", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )

        ch_a = self.back_a.feature_info.channels()
        ch_b = self.back_b.feature_info.channels()
        target_ch = [64, 96, 128, 160]

        self.proj_a = nn.ModuleList([nn.Conv2d(ca, tc, 1) for ca, tc in zip(ch_a, target_ch)])
        self.proj_b = nn.ModuleList([nn.Conv2d(cb, tc, 1) for cb, tc in zip(ch_b, target_ch)])
        self.fusions = nn.ModuleList([WeightedFusion(tc) for tc in target_ch])
        self.oams = nn.ModuleList([ObjectAwareAttention(tc) for tc in target_ch])

        self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)

        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._printed_debug = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats_a = self.back_a(x)
        feats_b = self.back_b(x)
        if not self._printed_debug:
            print("[WDFFNet] Backbone A feature channels:", [f.shape[1] for f in feats_a])
            print("[WDFFNet] Backbone B feature channels:", [f.shape[1] for f in feats_b])
            self._printed_debug = True
        fused_feats = []
        for pa, pb, fa, fb, fus, oam in zip(self.proj_a, self.proj_b, feats_a, feats_b, self.fusions, self.oams):
            f = fus(pa(fa), pb(fb))
            f = oam(f, None)
            fused_feats.append(f)

        f1, f2, f3, f4 = fused_feats
        cat3 = torch.cat([self.up3(f4), f3], dim=1)
        d3 = self.dec_conv3(cat3)
        cat2 = torch.cat([self.up2(d3), f2], dim=1)
        d2 = self.dec_conv2(cat2)
        cat1 = torch.cat([self.up1(d2), f1], dim=1)
        if cat3.shape[1] != 256 or cat2.shape[1] != 192 or cat1.shape[1] != 128:
            raise RuntimeError(
                f"WDFF decoder channel mismatch: cat3={cat3.shape}, cat2={cat2.shape}, cat1={cat1.shape}"
            )
        d1 = self.dec_conv1(cat1)
        return torch.sigmoid(self.out_conv(d1))


class BiFusionBlock(nn.Module):
    def __init__(self, cnn_ch: int, trans_ch: int, out_ch: int):
        super().__init__()
        self.conv_cnn = nn.Conv2d(cnn_ch, out_ch, kernel_size=1)
        self.conv_trans = nn.Conv2d(trans_ch, out_ch, kernel_size=1)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        if trans_feat.ndim == 4 and trans_feat.shape[1] < 10:
            trans_feat = trans_feat.permute(0, 3, 1, 2)
        x = self.conv_cnn(cnn_feat) + self.conv_trans(trans_feat)
        return self.act(self.conv_out(x))


class TransFuseSimple(nn.Module):
    """Architecture kept aligned with training notebook implementation."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, transfuse_input_size: int = 224):
        super().__init__()
        self.transfuse_input_size = transfuse_input_size
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True)
        self.trans = timm.create_model("mobilenetv3_large_100", pretrained=pretrained, features_only=True)

        cnn_ch = self.cnn.feature_info.channels()
        trans_ch = self.trans.feature_info.channels()

        self.fuse3 = BiFusionBlock(cnn_ch[-1], trans_ch[-1], 256)
        self.fuse2 = BiFusionBlock(cnn_ch[-2], trans_ch[-2], 128)
        self.fuse1 = BiFusionBlock(cnn_ch[-3], trans_ch[-3], 64)
        self.conv_final = nn.Conv2d(64 + 128 + 256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native_size = x.shape[2:]
        x_224 = F.interpolate(
            x, size=(self.transfuse_input_size, self.transfuse_input_size), mode="bilinear", align_corners=False
        )
        cnn_feats = self.cnn(x_224)
        trans_feats = []
        for t in self.trans(x_224):
            if t.ndim == 4 and t.shape[-1] > t.shape[1]:
                t = t.permute(0, 3, 1, 2)
            trans_feats.append(t)

        f3 = self.fuse3(cnn_feats[-1], trans_feats[-1])
        f2 = self.fuse2(cnn_feats[-2], trans_feats[-2])
        f1 = self.fuse1(cnn_feats[-3], trans_feats[-3])

        f3_up = F.interpolate(f3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        out = self.conv_final(torch.cat([f3_up, f2_up, f1], dim=1))
        out = torch.sigmoid(out)
        return F.interpolate(out, size=native_size, mode="bilinear", align_corners=False)


def import_resunetplusplus_builder(resunet_repo_dir: str):
    """Imports exact ResUNet++ builder from DebeshJha repository file."""
    path = os.path.join(resunet_repo_dir, "resunet++_pytorch.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ResUNet++ file not found: {path}")
    spec = importlib.util.spec_from_file_location("resunetplusplus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.build_resunetplusplus


# =========================
# Cell 4: Robust weight loader
# =========================

def _canonical_key(key: str) -> str:
    """Canonicalize parameter names to improve checkpoint/model key matching."""
    return key.replace("_", "").lower()


def load_partial_state_dict(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, int]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint: {ckpt_path}")
        return {"loaded": 0, "skipped": len(model.state_dict())}

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    model_dict = model.state_dict()
    clean_ckpt = {}
    for k, v in checkpoint.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "network."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        clean_ckpt[nk] = v

    loaded, skipped = [], []
    direct_hits = set()
    for k, v in clean_ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded.append(k)
            direct_hits.add(k)
        else:
            skipped.append(k)

    # Fuzzy fallback for known naming drifts (e.g. backA vs back_a, trans.* prefix changes).
    canon_to_model_keys: Dict[str, List[str]] = {}
    for mk in model_dict.keys():
        canon_to_model_keys.setdefault(_canonical_key(mk), []).append(mk)

    for ck, cv in clean_ckpt.items():
        if ck in direct_hits:
            continue
        candidates = canon_to_model_keys.get(_canonical_key(ck), [])
        for mk in candidates:
            if mk not in direct_hits and model_dict[mk].shape == cv.shape:
                model_dict[mk] = cv
                loaded.append(f"{ck} -> {mk}")
                direct_hits.add(mk)
                break

    model.load_state_dict(model_dict)
    missing_in_ckpt = [k for k in model_dict.keys() if k not in clean_ckpt]
    total_model_params = len(model_dict)
    loaded_ratio = len(loaded) / max(1, total_model_params)
    print(
        f"Loaded layers: {len(loaded)} | Skipped layers: {len(skipped)} | "
        f"Model params not found in checkpoint: {len(missing_in_ckpt)} | "
        f"Load ratio: {loaded_ratio:.1%}"
    )
    if skipped:
        print("Skipped keys (first 20):", skipped[:20])
    if missing_in_ckpt:
        print("Missing-in-ckpt keys (first 20):", missing_in_ckpt[:20])
    if loaded_ratio < 0.60:
        print(
            "[WARN] Low checkpoint load ratio detected (<60%). "
            "This usually means architecture-key mismatch and can severely hurt Dice."
        )
    return {"loaded": len(loaded), "skipped": len(skipped), "missing_in_ckpt": len(missing_in_ckpt)}


# =========================
# Cell 5: Dataset / DataLoader
# =========================

class KvasirDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 352, augment: bool = False):
        self.df = df.reset_index(drop=True)
        if augment:
            self.tf = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.tf = A.Compose([A.Resize(size, size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, -1)
        out = self.tf(image=img, mask=mask)
        x = out["image"].astype("float32").transpose(2, 0, 1)
        y = (out["mask"].astype("float32") / 255.0)
        y = np.clip(y, 0.0, 1.0)
        y = (y > 0.5).astype("float32").transpose(2, 0, 1)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(data_dir: str, size: int = 352, batch_size: int = 8, num_workers: int = 2):
    images = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))
    if len(masks) == 0:
        masks = sorted(glob(os.path.join(data_dir, "masks", "*.png")))
    assert len(images) == len(masks), "Image/mask count mismatch"

    df = pd.DataFrame({"image": images, "mask": masks}).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = KvasirDataset(train_df, size=size, augment=True)
    val_ds = KvasirDataset(val_df, size=size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# =========================
# Cell 6: HAR Ensemble
# =========================

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        avg_map = torch.mean(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([max_map, avg_map], dim=1))


class HAREnsemble(nn.Module):
    def __init__(self, resunetpp: nn.Module, wdff: nn.Module, transfuse: nn.Module):
        super().__init__()
        self.resunetpp = resunetpp.eval()
        self.wdff = wdff.eval()
        self.transfuse = transfuse.eval()
        # Backward-compatible aliases used by some notebook cells.
        self.r = self.resunetpp
        self.w = self.wdff
        self.t = self.transfuse

        for m in [self.resunetpp, self.wdff, self.transfuse]:
            for p in m.parameters():
                p.requires_grad = False

        self.attn = ChannelSpatialAttention(in_channels=3)
        self.weight_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.Softmax(dim=1),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self._printed_debug = False

    def _align(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return normalize_segmentation_output(pred, ref_shape=ref.shape[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            p1 = self.resunetpp(x)
            p2 = self.wdff(x)
            p3 = self.transfuse(x)

        p1 = self._align(p1, x)
        p2 = self._align(p2, x)
        p3 = self._align(p3, x)
        if not self._printed_debug:
            print(f"[HAR] Shapes -> p1:{tuple(p1.shape)} p2:{tuple(p2.shape)} p3:{tuple(p3.shape)}")
            self._printed_debug = True

        stack = torch.cat([p1, p2, p3], dim=1)
        stack = self.attn(stack)
        weights = self.weight_head(stack)
        wr, wt, ww = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = wr * p1 + wt * p3 + ww * p2
        fused = fused * self.spatial_attention(stack)
        return fused


class RefinementModel(nn.Module):
    """Small UNet-like second stage that refines ensemble predictions."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


# =========================
# Cell 7: Metrics + evaluation
# =========================



def normalize_segmentation_output(pred: torch.Tensor, ref_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Normalize model output to BCHW tensor and optionally resize to ref_shape (H, W)."""
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            raise ValueError("Model output is empty list/tuple.")
        pred = pred[0]

    if pred.ndim == 2:
        # HW -> BCHW
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    elif pred.ndim == 4 and pred.shape[-1] in (1, 2, 3) and pred.shape[1] not in (1, 2, 3):
        # NHWC -> NCHW (more robust condition)
        pred = pred.permute(0, 3, 1, 2)
    elif pred.ndim != 4:
        raise ValueError(f"Unsupported prediction shape: {tuple(pred.shape)}")

    # If model accidentally returns multi-channel logits, keep first channel for binary segmentation.
    if pred.shape[1] > 1:
        pred = pred[:, :1, ...]

    if ref_shape is not None and pred.shape[2:] != ref_shape:
        pred = F.interpolate(pred, size=ref_shape, mode="bilinear", align_corners=False)
    # Some checkpoints/models return logits. Convert to probabilities consistently.
    if pred.min().item() < 0.0 or pred.max().item() > 1.0:
        pred = torch.sigmoid(pred)
    return pred

def _bin(pred: torch.Tensor, th: float = 0.5) -> torch.Tensor:
    return (pred > th).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = _bin(pred, threshold).reshape(-1)
    target = _bin(target).reshape(-1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = _bin(pred, threshold).reshape(-1)
    target = _bin(target).reshape(-1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return float((tp + eps) / (tp + fp + fn + eps))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = _bin(pred, th=threshold)
    target = _bin(target)
    pred_f = pred.reshape(-1)
    tar_f = target.reshape(-1)

    tp = ((pred_f == 1) & (tar_f == 1)).sum().item()
    fp = ((pred_f == 1) & (tar_f == 0)).sum().item()
    fn = ((pred_f == 0) & (tar_f == 1)).sum().item()
    tn = ((pred_f == 0) & (tar_f == 0)).sum().item()

    dice = dice_score(pred, target, threshold=0.5)
    iou = iou_score(pred, target, threshold=0.5)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    return {"Dice": dice, "IoU": iou, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    use_postprocess: bool = False,
    use_tta: bool = False,
) -> Dict[str, float]:
    model.eval()
    agg = {"Dice": [], "IoU": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_tta:
            p = tta_predict(model, x)
        else:
            p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
        if use_postprocess:
            p_np = p.detach().cpu().numpy()
            pp = []
            for bi in range(p_np.shape[0]):
                proc = post_process_mask((p_np[bi, 0] > threshold).astype(np.uint8))
                pp.append(torch.from_numpy((proc > 0).astype(np.float32))[None, ...])
            p = torch.stack(pp, dim=0).to(device)
        batch_metrics = compute_metrics(p, y, threshold=threshold)
        for k, v in batch_metrics.items():
            agg[k].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


@torch.no_grad()
def collect_preds_and_masks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, masks = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_tta:
            p = tta_predict(model, x)
        else:
            p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
        preds.append(p.detach().cpu())
        masks.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(masks, dim=0)


def find_best_threshold(preds: torch.Tensor, masks: torch.Tensor) -> float:
    """Search thresholds from 0.20 to 0.60 (step 0.05) and pick best Dice."""
    candidates = np.arange(0.20, 0.61, 0.05)
    best_t, best_dice = 0.5, -1.0
    for t in candidates:
        d = dice_score(preds, masks, threshold=float(t))
        if d > best_dice:
            best_dice = d
            best_t = float(t)
    print(f"[HAR] Best validation threshold: {best_t:.2f} (Dice={best_dice:.4f})")
    return best_t


def multi_scale_predict(model: nn.Module, img: torch.Tensor, scales: Iterable[float] = (0.75, 1.0, 1.25)) -> torch.Tensor:
    """Run model at multiple scales and return averaged sigmoid output."""
    model.eval()
    _, _, h, w = img.shape
    preds = []
    for s in scales:
        nh = max(16, int(round(h * s)))
        nw = max(16, int(round(w * s)))
        scaled = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
        pred = normalize_segmentation_output(model(scaled), ref_shape=(nh, nw))
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        preds.append(pred)
    return torch.mean(torch.stack(preds, dim=0), dim=0)


def tta_predict(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """TTA with horizontal/vertical flips, each branch using multi-scale inference."""
    model.eval()
    preds = [multi_scale_predict(model, img)]
    fh = torch.flip(img, dims=[3])
    preds.append(torch.flip(multi_scale_predict(model, fh), dims=[3]))
    fv = torch.flip(img, dims=[2])
    preds.append(torch.flip(multi_scale_predict(model, fv), dims=[2]))
    return torch.mean(torch.stack(preds, dim=0), dim=0)


# =========================
# Cell 8: Train only HAR head
# =========================

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    inter = (pred_f * target_f).sum(dim=1)
    union = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.8, gamma: float = 2.0) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boundary-aware loss via Laplacian-like gradient mismatch."""
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    pred_b = torch.abs(F.conv2d(pred, kernel, padding=1))
    tar_b = torch.abs(F.conv2d(target, kernel, padding=1))
    return F.l1_loss(pred_b, tar_b)


def combined_seg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss_bce = F.binary_cross_entropy(pred.float(), target.float())
    loss_dice = dice_loss(pred, target)
    loss_focal = focal_loss(pred, target)
    loss_boundary = boundary_loss(pred, target)
    return 0.3 * loss_bce + 0.3 * loss_dice + 0.3 * loss_focal + 0.1 * loss_boundary


def train_har_head(model: HAREnsemble, train_loader: DataLoader, cfg: TrainConfig):
    optimizer = torch.optim.Adam(
        list(model.attn.parameters()) + list(model.weight_head.parameters()) + list(model.spatial_attention.parameters()),
        lr=cfg.lr,
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    model.to(DEVICE)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=USE_AMP):
                p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
                loss = combined_seg_loss(p, y)
                batch_dice = dice_score(p.detach(), y.detach(), threshold=0.5)
                if batch_dice < 0.7:
                    loss = loss * 1.5

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        print(f"Epoch {ep}/{cfg.epochs} | HAR head loss: {run_loss / len(train_loader):.4f}")


def train_refinement_model(
    refine_model: RefinementModel,
    ensemble_model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
) -> None:
    refine_model.to(DEVICE)
    ensemble_model.to(DEVICE).eval()
    optimizer = torch.optim.Adam(refine_model.parameters(), lr=max(5e-4, cfg.lr * 0.5))
    scaler = GradScaler("cuda", enabled=USE_AMP)

    for ep in range(1, cfg.epochs + 1):
        refine_model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                p1 = normalize_segmentation_output(ensemble_model(x), ref_shape=y.shape[2:])
            with autocast(device_type="cuda", enabled=USE_AMP):
                ref_input = torch.cat([x, p1], dim=1)
                p2 = normalize_segmentation_output(refine_model(ref_input), ref_shape=y.shape[2:])
                loss = combined_seg_loss(p2, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        print(f"Epoch {ep}/{cfg.epochs} | Refinement loss: {run_loss / len(train_loader):.4f}")


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    model_name: str = "Model",
) -> None:
    """Uniform base-model fine-tuning loop used for fair comparison."""
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler("cuda", enabled=USE_AMP)
    for ep in range(1, cfg.epochs + 1):
        run_loss = 0.0
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=USE_AMP):
                p = normalize_segmentation_output(model(x), ref_shape=y.shape[2:])
                loss = combined_seg_loss(p, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        print(f"[{model_name}] Epoch {ep}/{cfg.epochs} | loss: {run_loss / len(train_loader):.4f}")


def save_model_checkpoint(model: nn.Module, name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{name.lower().replace('+', 'plus').replace(' ', '_')}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def save_run_hyperparams(params: Dict[str, object], output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hp_df = pd.DataFrame([params])
    out = output_dir / "run_hyperparameters.csv"
    hp_df.to_csv(out, index=False)
    return out


# =========================
# Cell 9: End-to-end run
# =========================

# Example paths for Google Drive checkpoints:
# resunet_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_resunetpp_model.pth"
# wdff_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_wdffnet_model.pth"  # optional
# transfuse_ckpt = "/content/drive/MyDrive/BalkuProject_Outputs/best_transfuse_model.pth"

# Example:
# RESUNET_REPO = "/content/ResUNetPlusPlus"
# clone_repo_if_needed("https://github.com/DebeshJha/ResUNetPlusPlus.git", RESUNET_REPO)
# ResUnetPPBuilder = import_resunetplusplus_builder(RESUNET_REPO)
# resunetpp = ResUnetPPBuilder().to(DEVICE)
# wdffnet = WDFFNet(pretrained=False, num_classes=1).to(DEVICE)
# transfuse = TransFuseSimple(num_classes=1, pretrained=False).to(DEVICE)
#
# load_partial_state_dict(resunetpp, resunet_ckpt, DEVICE)
# load_partial_state_dict(wdffnet, wdff_ckpt, DEVICE)        # handled gracefully if missing
# load_partial_state_dict(transfuse, transfuse_ckpt, DEVICE)
#
# train_loader, val_loader = make_loaders("/content/data/Kvasir-SEG", size=352, batch_size=8, num_workers=2)
# har = HAREnsemble(resunetpp, wdffnet, transfuse).to(DEVICE)
# train_har_head(har, train_loader, TrainConfig(epochs=12, lr=1e-3))
#
# base_wrappers = {
#     "ResUNet++": resunetpp,
#     "WDFFNet": wdffnet,
#     "TransFuse": transfuse,
#     "HAR Ensemble": har,
# }
#
# rows = []
# for name, mdl in base_wrappers.items():
#     m = evaluate_model(mdl, val_loader, DEVICE)
#     rows.append([name, m["Dice"], m["IoU"], m["Precision"], m["Recall"], m["F1"]])
#
# print(tabulate(rows, headers=["Model", "Dice", "IoU", "Precision", "Recall", "F1"], floatfmt=".4f", tablefmt="github"))


def _env_path(name: str) -> Optional[str]:
    raw = os.environ.get(name, "").strip()
    return raw if raw else None


def _must_exist(path: Optional[str], label: str) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _auto_prepare_data_dir() -> Optional[str]:
    """
    Resolve dataset path more robustly for Colab.
    Priority:
      1) DATA_DIR if exists
      2) Extract DATA_ZIP (if provided) to /content/data/
      3) Check common Colab/Drive fallback locations
    """
    data_dir = _env_path("DATA_DIR")
    if data_dir and os.path.exists(data_dir):
        return data_dir

    data_zip = _env_path("DATA_ZIP")
    if data_zip and os.path.exists(data_zip):
        target_root = Path("/content/data")
        target_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Extracting dataset zip: {data_zip} -> {target_root}")
        subprocess.run(["unzip", "-o", data_zip, "-d", str(target_root)], check=True)
        # Try common extracted folder names.
        candidates = [
            target_root / "Kvasir-SEG",
            target_root / "kvasir-seg",
            target_root / "Kvasir_SEG",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    fallback_candidates = [
        "/content/data/Kvasir-SEG",
        "/content/Kvasir-SEG",
        "/content/drive/MyDrive/Kvasir-SEG",
        "/content/drive/MyDrive/data/Kvasir-SEG",
    ]
    for c in fallback_candidates:
        if os.path.exists(c):
            print(f"[INFO] Using fallback DATA_DIR: {c}")
            return c
    return None


def _auto_prepare_resunet_repo() -> Optional[str]:
    """Resolve or optionally clone ResUNet++ repository in Colab."""
    repo = _env_path("RESUNET_REPO")
    if repo and os.path.exists(repo):
        return repo

    repo_url = _env_path("RESUNET_REPO_URL") or "https://github.com/DebeshJha/ResUNetPlusPlus.git"
    default_repo = Path("/content/ResUNetPlusPlus")
    if not default_repo.exists():
        print(f"[INFO] Cloning ResUNet++ repo from: {repo_url}")
        clone_repo_if_needed(repo_url, str(default_repo))
    if default_repo.exists():
        return str(default_repo)
    return None


def mount_drive_if_needed() -> None:
    """Mount Google Drive only when running inside Colab and not already mounted."""
    in_colab = importlib.util.find_spec("google.colab") is not None
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return
    if in_colab:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")


def post_process_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Median blur + closing + small component removal."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    mask_u8 = cv2.medianBlur(mask_u8, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def save_plot(fig, name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def visualize_predictions(model: nn.Module, loader: DataLoader, output_dir: Path = OUTPUT_DIR, num_samples: int = 4) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    shown = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = tta_predict(model, x)
            for bi in range(x.size(0)):
                if shown >= num_samples:
                    return
                img = x[bi].detach().cpu().permute(1, 2, 0).numpy()
                img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
                gt = y[bi, 0].detach().cpu().numpy()
                pr = (pred[bi, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                pr = post_process_mask(pr)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[1].imshow(gt, cmap="gray")
                axes[1].set_title("GT")
                axes[2].imshow(pr, cmap="gray")
                axes[2].set_title("Pred")
                for ax in axes:
                    ax.axis("off")
                save_plot(fig, f"prediction_{shown:03d}.png", output_dir=output_dir)
                plt.close(fig)
                shown += 1


def plot_metrics(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    import matplotlib.pyplot as plt

    for metric in ["Dice", "IoU"]:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(df["Model"], df[metric], marker="o")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        plt.title(f"{metric} Comparison")
        save_plot(fig, f"{metric.lower()}_plot.png", output_dir=output_dir)
        plt.close(fig)


@torch.no_grad()
def analyze_models_on_images(
    models: Dict[str, nn.Module],
    image_paths: List[str],
    output_file: str = "model_comparison_samples.png",
    output_dir: Path = OUTPUT_DIR,
    image_size: int = 320,
    threshold: float = 0.5,
) -> Path:
    """
    Analyze 1-2 images with ResUNet++, TransFuse, WDFFNet, and Ensemble in one file.
    Produces a single comparison sheet: rows=samples, cols=[Image, each model prediction].
    """
    import matplotlib.pyplot as plt

    if not image_paths:
        raise ValueError("image_paths cannot be empty.")
    image_paths = image_paths[:2]  # user requested one or two samples
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(models.keys())
    for m in models.values():
        m.to(DEVICE).eval()

    tf = A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    nrows = len(image_paths)
    ncols = 1 + len(model_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, img_path in enumerate(image_paths):
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        aug = tf(image=rgb)
        x_np = aug["image"].astype("float32").transpose(2, 0, 1)
        x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"Input #{r + 1}")
        axes[r, 0].axis("off")

        for c, model_name in enumerate(model_names, start=1):
            model = models[model_name]
            pred = tta_predict(model, x) if model_name.lower().startswith("har") else multi_scale_predict(model, x)
            pred_np = pred[0, 0].detach().cpu().numpy()
            mask = (pred_np > threshold).astype(np.uint8)
            mask = post_process_mask(mask)
            axes[r, c].imshow(mask, cmap="gray")
            axes[r, c].set_title(model_name)
            axes[r, c].axis("off")

    fig.tight_layout()
    out_path = output_dir / output_file
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] model comparison sheet: {out_path}")
    return out_path


@torch.no_grad()
def save_loader_comparison_sheet(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    output_file: str = "all_models_visual_comparison.png",
    output_dir: Path = OUTPUT_DIR,
    num_samples: int = 6,
    threshold: float = 0.5,
) -> Path:
    """
    Save a single image sheet with Image + GT + all model masks for easy visual comparison.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for m in models.values():
        m.to(DEVICE).eval()

    cols = ["Image", "GT"] + list(models.keys())
    ncols = len(cols)
    nrows = num_samples
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    written = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y_np = y.numpy()
        preds_by_model: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            pred = tta_predict(model, x) if "HAR" in name else multi_scale_predict(model, x)
            preds_by_model[name] = pred.detach().cpu().numpy()

        for bi in range(x.size(0)):
            if written >= num_samples:
                break
            img = x[bi].detach().cpu().permute(1, 2, 0).numpy()
            img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
            gt = y_np[bi, 0]

            axes[written, 0].imshow(img)
            axes[written, 0].set_title("Image")
            axes[written, 1].imshow(gt, cmap="gray")
            axes[written, 1].set_title("GT")
            axes[written, 0].axis("off")
            axes[written, 1].axis("off")

            ci = 2
            for name in models.keys():
                mask = (preds_by_model[name][bi, 0] > threshold).astype(np.uint8)
                mask = post_process_mask(mask)
                axes[written, ci].imshow(mask, cmap="gray")
                axes[written, ci].set_title(name)
                axes[written, ci].axis("off")
                ci += 1
            written += 1
        if written >= num_samples:
            break

    fig.tight_layout()
    out = output_dir / output_file
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] loader comparison sheet: {out}")
    return out


@torch.no_grad()
def estimate_fps(model: nn.Module, loader: DataLoader, warmup: int = 3, measured_batches: int = 10) -> float:
    model.eval()
    iterator = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        _ = model(x.to(DEVICE))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    seen = 0
    start = time.time()
    for _ in range(measured_batches):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        _ = model(x.to(DEVICE))
        seen += x.size(0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.time() - start)
    return seen / elapsed


def run_full_pipeline() -> None:
    """
    Execute full HAR pipeline using environment variables.

    Required:
      DATA_DIR, RESUNET_REPO, RESUNET_CKPT, TRANSFUSE_CKPT
    Optional:
      WDFF_CKPT, EPOCHS, BATCH_SIZE, IMG_SIZE, FINETUNE_SIZE, NUM_WORKERS
    """
    mount_drive_if_needed()

    data_dir = _must_exist(_auto_prepare_data_dir(), "DATA_DIR")
    resunet_repo = _must_exist(_auto_prepare_resunet_repo(), "RESUNET_REPO")
    resunet_ckpt = _must_exist(_env_path("RESUNET_CKPT"), "RESUNET_CKPT")
    transfuse_ckpt = _must_exist(_env_path("TRANSFUSE_CKPT"), "TRANSFUSE_CKPT")
    wdff_ckpt = _env_path("WDFF_CKPT")
    if wdff_ckpt and not os.path.exists(wdff_ckpt):
        print(f"[WARN] WDFF_CKPT not found: {wdff_ckpt}. WDFFNet will run with random weights.")
        wdff_ckpt = None

    missing = [
        name
        for name, value in [
            ("DATA_DIR", data_dir),
            ("RESUNET_REPO", resunet_repo),
            ("RESUNET_CKPT", resunet_ckpt),
            ("TRANSFUSE_CKPT", transfuse_ckpt),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nSet them before running `%run colab_har_ensemble.py`.\n"
            + "Tips:\n"
            + "  - Set DATA_DIR directly, or set DATA_ZIP to a zip path in Drive.\n"
            + "  - Set RESUNET_REPO directly, or optionally set RESUNET_REPO_URL for auto-clone."
        )

    epochs = int(os.environ.get("EPOCHS", "12"))
    base_epochs = int(os.environ.get("BASE_EPOCHS", "2"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    img_size = int(os.environ.get("IMG_SIZE", "256"))
    finetune_size = int(os.environ.get("FINETUNE_SIZE", "320"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_hyperparams(
        {
            "epochs": epochs,
            "base_epochs": base_epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "finetune_size": finetune_size,
            "num_workers": num_workers,
            "device": str(DEVICE),
            "amp": USE_AMP,
        },
        output_dir=output_dir,
    )

    ResUnetPPBuilder = import_resunetplusplus_builder(resunet_repo)
    resunetpp = ResUnetPPBuilder().to(DEVICE)
    wdffnet = WDFFNet(pretrained=False, num_classes=1).to(DEVICE)
    transfuse = TransFuseSimple(num_classes=1, pretrained=False, transfuse_input_size=224).to(DEVICE)

    load_partial_state_dict(resunetpp, resunet_ckpt, DEVICE)
    load_partial_state_dict(wdffnet, wdff_ckpt or "", DEVICE)
    load_partial_state_dict(transfuse, transfuse_ckpt, DEVICE)

    train_loader, val_loader = make_loaders(
        data_dir,
        size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Optional high-resolution fine-tune loaders.
    train_loader_ft, val_loader_ft = make_loaders(
        data_dir,
        size=finetune_size,
        batch_size=max(1, batch_size // 2),
        num_workers=num_workers,
    )
    # 1) Uniform base-model training before ensemble training.
    if base_epochs > 0:
        train_single_model(resunetpp, train_loader, TrainConfig(epochs=base_epochs, lr=8e-5), model_name="ResUNet++")
        train_single_model(transfuse, train_loader, TrainConfig(epochs=base_epochs, lr=8e-5), model_name="TransFuse")
        train_single_model(wdffnet, train_loader, TrainConfig(epochs=base_epochs, lr=8e-5), model_name="WDFFNet")

    har = HAREnsemble(resunetpp, wdffnet, transfuse).to(DEVICE)
    # 2) Ensemble training.
    train_har_head(har, train_loader, TrainConfig(epochs=epochs, lr=1e-3))
    if finetune_size > img_size:
        train_har_head(har, train_loader_ft, TrainConfig(epochs=max(2, epochs // 3), lr=5e-4))

    # 3) Refinement training.
    refine_model = RefinementModel().to(DEVICE)
    train_refinement_model(refine_model, har, train_loader_ft, TrainConfig(epochs=max(3, epochs // 2), lr=8e-4))

    class CascadeWrapper(nn.Module):
        def __init__(self, ens: nn.Module, ref: nn.Module):
            super().__init__()
            self.ens = ens
            self.ref = ref

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            p1 = normalize_segmentation_output(self.ens(x), ref_shape=x.shape[2:])
            p2 = self.ref(torch.cat([x, p1], dim=1))
            return normalize_segmentation_output(p2, ref_shape=x.shape[2:])

    cascade_model = CascadeWrapper(har, refine_model).to(DEVICE)
    val_preds, val_masks = collect_preds_and_masks(cascade_model, val_loader_ft, DEVICE, use_tta=True)
    best_thr = find_best_threshold(val_preds, val_masks)

    base_wrappers = {
        "ResUNet++": resunetpp,
        "WDFFNet": wdffnet,
        "TransFuse": transfuse,
        "HAR Ensemble": har,
        "HAR+Refine": cascade_model,
    }

    rows, csv_rows = [], []
    for name, mdl in base_wrappers.items():
        threshold = best_thr if name in ("HAR Ensemble", "HAR+Refine") else 0.5
        m = evaluate_model(
            mdl,
            val_loader_ft,
            DEVICE,
            threshold=threshold,
            use_postprocess=(name == "HAR+Refine"),
            use_tta=(name == "HAR+Refine"),
        )
        params = sum(p.numel() for p in mdl.parameters()) / 1e6
        fps = estimate_fps(mdl, val_loader_ft)
        rows.append([name, m["Dice"], m["IoU"], params, fps])
        csv_rows.append({"Model": name, "Dice": m["Dice"], "IoU": m["IoU"], "Params(M)": params, "FPS": fps})
        ckpt = save_model_checkpoint(mdl, name, output_dir=output_dir)
        print(f"[Saved] {name} checkpoint -> {ckpt}")

    print(
        tabulate(
            rows,
            headers=["Model", "Dice", "IoU", "Params(M)", "FPS"],
            floatfmt=".4f",
            tablefmt="github",
        )
    )
    comp_df = pd.DataFrame(csv_rows)
    comp_df.to_csv(output_dir / "comparison_table.csv", index=False)
    plot_metrics(comp_df, output_dir=output_dir)
    visualize_predictions(cascade_model, val_loader_ft, output_dir=output_dir, num_samples=6)
    save_loader_comparison_sheet(
        models=base_wrappers,
        loader=val_loader_ft,
        output_file="all_models_visual_comparison.png",
        output_dir=output_dir,
        num_samples=6,
        threshold=best_thr,
    )
    # Optional: compare 1-2 specific images across base models + ensemble in one file.
    # Example:
    #   %env ANALYZE_IMAGES=/content/data/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg,/content/data/Kvasir-SEG/images/xxx.jpg
    analyze_images = os.environ.get("ANALYZE_IMAGES", "").strip()
    if analyze_images:
        selected = [p.strip() for p in analyze_images.split(",") if p.strip() and os.path.exists(p.strip())][:2]
        if selected:
            analyze_models_on_images(
                models={
                    "ResUNet++": resunetpp,
                    "TransFuse": transfuse,
                    "WDFFNet": wdffnet,
                    "HAR Ensemble": har,
                },
                image_paths=selected,
                output_file="studied_models_comparison.png",
                output_dir=output_dir,
                image_size=finetune_size,
                threshold=best_thr,
            )


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as exc:
        print(f"[INFO] End-to-end run skipped: {exc}")
        if "device-side assert" in str(exc).lower():
            print(
                "[INFO] CUDA device-side assert detected. Re-run with:\n"
                "  %env CUDA_LAUNCH_BLOCKING=1\n"
                "  %env DISABLE_AMP=1\n"
                "or force CPU for debugging:\n"
                "  %env FORCE_CPU=1"
            )
        print(
            "[INFO] To run in Colab, set env vars first:\n"
            "  %env DATA_DIR=/content/data/Kvasir-SEG\n"
            "  # or provide zipped dataset:\n"
            "  %env DATA_ZIP=/content/drive/MyDrive/datasets/Kvasir-SEG.zip\n"
            "  %env RESUNET_REPO=/content/ResUNetPlusPlus\n"
            "  # optional for auto-clone:\n"
            "  %env RESUNET_REPO_URL=https://github.com/DebeshJha/ResUNetPlusPlus.git\n"
            "  %env RESUNET_CKPT=/content/drive/MyDrive/.../best_resunetpp_model.pth\n"
            "  %env WDFF_CKPT=/content/drive/MyDrive/.../best_wdffnet.pth\n"
            "  %env TRANSFUSE_CKPT=/content/drive/MyDrive/.../best_transfuse_model.pth\n"
            "  %run colab_har_ensemble.py"
        )

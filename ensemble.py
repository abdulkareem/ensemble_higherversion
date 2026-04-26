from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_metrics_from_logits, dice_bce_loss, ensure_binary_output


class CrossScaleGatedFusionModule(nn.Module):
    """Cross-Scale Gated Fusion for robust small/large polyp handling.

    Inputs are branch logits stacked as (B, 3, H, W) in the order:
    [VM-UNet(Mamba), TransFuse(Transformer), ResUNet++(CNN)].
    """

    def __init__(self, in_branches: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.in_branches = in_branches

        # Learns pixel-wise scale preferences (fine/mid/coarse).
        self.scale_gate = nn.Sequential(
            nn.Conv2d(in_branches * 3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

        # Learns pixel-wise branch preferences after scale aggregation.
        self.branch_gate = nn.Sequential(
            nn.Conv2d(in_branches, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_branches, kernel_size=1),
        )

        # Optional residual refinement for sharper boundaries.
        self.refine = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _scale_pyramid(self, stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = stack
        mid = self._resize_like(F.avg_pool2d(stack, kernel_size=2, stride=2), stack)
        coarse = self._resize_like(F.avg_pool2d(stack, kernel_size=4, stride=4), stack)
        return fine, mid, coarse

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        fine, mid, coarse = self._scale_pyramid(stack)

        scale_logits = self.scale_gate(torch.cat([fine, mid, coarse], dim=1))
        scale_weights = torch.softmax(scale_logits, dim=1)

        scale_agg = (
            scale_weights[:, 0:1] * fine
            + scale_weights[:, 1:2] * mid
            + scale_weights[:, 2:3] * coarse
        )

        branch_logits = self.branch_gate(scale_agg)
        branch_weights = torch.softmax(branch_logits, dim=1)
        fused = (branch_weights * scale_agg).sum(dim=1, keepdim=True)
        return fused + self.refine(fused)


class MambaFusionEnsemble(nn.Module):
    def __init__(self, vm_unet_mamba: nn.Module, transfuse: nn.Module, resunetpp: nn.Module):
        super().__init__()
        self.mamba = vm_unet_mamba.eval()
        self.transfuse = transfuse.eval()
        self.resunetpp = resunetpp.eval()

        for m in [self.mamba, self.transfuse, self.resunetpp]:
            for p in m.parameters():
                p.requires_grad = False

        self.cross_scale_fusion = CrossScaleGatedFusionModule(in_branches=3, hidden_channels=32)

    def forward_branch_logits(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            m = ensure_binary_output(self.mamba(x))
            t = ensure_binary_output(self.transfuse(x))
            r = ensure_binary_output(self.resunetpp(x))
        return {"mamba": m, "transformer": t, "cnn": r}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = self.forward_branch_logits(x)
        stack = torch.cat([branches["mamba"], branches["transformer"], branches["cnn"]], dim=1)
        return self.cross_scale_fusion(stack)


# Backward-compatible alias for older scripts.
WeightedEnsemble = MambaFusionEnsemble


def train_ensemble_head(
    model: MambaFusionEnsemble,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.cross_scale_fusion.parameters(), lr=lr)
    history: Dict[str, list] = {"train_loss": [], "val_dice": []}
    best_dice = -1.0
    best_path = "best_mamba_fusion_ensemble.pth"

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = dice_bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                m = compute_metrics_from_logits(model(x), y)
                dices.append(m["Dice"])

        train_loss = float(np.mean(losses))
        val_dice = float(np.mean(dices))
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        print(f"[Mamba-Fusion] Epoch {ep}/{epochs} | loss={train_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return history, best_path

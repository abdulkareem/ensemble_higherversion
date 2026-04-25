from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_metrics_from_logits, dice_bce_loss, ensure_binary_output


class CrossScaleFusionModule(nn.Module):
    """Adaptive multi-scale fusion to address polyp size variation."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16):
        super().__init__()
        self.scale_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1)
        self.scale_2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.scale_3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=3, dilation=3)
        self.reweight = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        s1 = F.relu(self.scale_1(stack), inplace=True)
        s2 = F.relu(self.scale_2(stack), inplace=True)
        s3 = F.relu(self.scale_3(stack), inplace=True)

        multi = torch.cat([s1, s2, s3], dim=1)
        alpha = torch.softmax(self.reweight(multi), dim=1)
        fused = (alpha * stack).sum(dim=1, keepdim=True)
        return fused


class MambaFusionEnsemble(nn.Module):
    def __init__(self, vm_unet_mamba: nn.Module, transfuse: nn.Module, resunetpp: nn.Module):
        super().__init__()
        self.mamba = vm_unet_mamba.eval()
        self.transfuse = transfuse.eval()
        self.resunetpp = resunetpp.eval()

        for m in [self.mamba, self.transfuse, self.resunetpp]:
            for p in m.parameters():
                p.requires_grad = False

        self.cross_scale_fusion = CrossScaleFusionModule(in_channels=3, hidden_channels=24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = ensure_binary_output(self.mamba(x))
            t = ensure_binary_output(self.transfuse(x))
            r = ensure_binary_output(self.resunetpp(x))

        stack = torch.cat([m, t, r], dim=1)
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

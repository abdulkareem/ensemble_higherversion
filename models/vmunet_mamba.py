import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaStyleBlock(nn.Module):
    """Lightweight state-space inspired block for VM-UNet style modeling."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.pw_in = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pw_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.pw_in(x)
        gate, value = torch.chunk(x, chunks=2, dim=1)
        x = torch.sigmoid(gate) * self.act(value)
        x = self.pw_out(x)
        return x + residual


class VMUNetMamba(nn.Module):
    def __init__(self, out_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        ch = self.encoder.feature_info.channels()
        target_ch = [64, 96, 128, 160]

        self.proj = nn.ModuleList([nn.Conv2d(cin, cout, kernel_size=1) for cin, cout in zip(ch, target_ch)])
        self.mamba = nn.ModuleList([MambaStyleBlock(c) for c in target_ch])

        self.up3 = nn.ConvTranspose2d(160, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 96, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)

        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(96 + 96, 96, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.out_head = nn.Conv2d(64, 1, kernel_size=1)
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        f1, f2, f3, f4 = [m(p(f)) for p, m, f in zip(self.proj, self.mamba, feats)]

        d3 = self.dec3(torch.cat([self.up3(f4), f3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))

        logits = self.out_head(d1)
        if logits.shape[-2:] != (self.out_size, self.out_size):
            logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return logits

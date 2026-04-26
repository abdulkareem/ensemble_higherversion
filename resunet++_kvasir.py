import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ASPP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.b1 = nn.Conv2d(channels, channels, 1)
        self.b2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.proj = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.proj(out)


class ResUNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualConv(3, 64)
        self.enc2 = ResidualConv(64, 96)
        self.enc3 = ResidualConv(96, 128)
        self.enc4 = ResidualConv(128, 160)

        self.pool = nn.MaxPool2d(2, 2)
        self.aspp = ASPP(160)

        self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)

        self.dec3 = ResidualConv(128 + 128, 128)
        self.dec2 = ResidualConv(96 + 96, 96)
        self.dec1 = ResidualConv(64 + 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.aspp(e4)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


def build_resunetplusplus() -> nn.Module:
    return ResUNetPlusPlus()

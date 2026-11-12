from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetTokenEncoder(nn.Module):
    """A lightweight CNN token encoder without torchvision dependency."""
    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.stage1 = ConvBlock(in_channels, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)
        self.stage4 = ConvBlock(128, 256, stride=2)
        self.token_proj = nn.Linear(256, embed_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.stage4(x)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        pooled = F.adaptive_avg_pool2d(feat, 1).view(B, C)
        pooled = self.global_proj(pooled)
        return tokens, pooled, (H, W)

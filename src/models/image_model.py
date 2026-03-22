from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead
from models.image_encoder import ResNetTokenEncoder


class ImageOnlyModel(nn.Module):
    def __init__(self, image_embed_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.image_encoder = ResNetTokenEncoder(in_channels=1, embed_dim=image_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = CORNHead(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, y: torch.Tensor | None = None):
        _tokens, image_global, grid_hw = self.image_encoder(image)
        z = self.fusion(image_global)
        out = self.head(z, y)
        out["grid_hw"] = grid_hw
        out["fusion_out"] = z
        return out
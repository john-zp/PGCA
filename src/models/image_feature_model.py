from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead
from models.image_encoder import ResNetTokenEncoder


class ImageFeatureModel(nn.Module):
    def __init__(self, feature_dim: int, image_embed_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.image_encoder = ResNetTokenEncoder(in_channels=1, embed_dim=image_embed_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = CORNHead(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, features: torch.Tensor, y: torch.Tensor | None = None):
        _tokens, image_global, grid_hw = self.image_encoder(image)
        feature_vec = self.feature_encoder(features)
        z = self.fusion(torch.cat([image_global, feature_vec], dim=1))
        out = self.head(z, y)
        out["grid_hw"] = grid_hw
        out["fusion_out"] = z
        return out
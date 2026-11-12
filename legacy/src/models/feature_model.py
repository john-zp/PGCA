from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead


class FeatureOnlyModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: list[int], dropout: float, num_classes: int):
        super().__init__()
        dims = [feature_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = CORNHead(dims[-1], num_classes)

    def forward(self, features: torch.Tensor, y: torch.Tensor | None = None):
        z = self.backbone(features)
        out = self.head(z, y)
        out["features_out"] = z
        return out

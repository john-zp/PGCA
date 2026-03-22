from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead


class PhraseOnlyModel(nn.Module):
    def __init__(self, phrase_dim: int, hidden_dims: list[int], dropout: float, num_classes: int):
        super().__init__()
        dims = [phrase_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = CORNHead(dims[-1], num_classes)

    def forward(self, phrase_emb: torch.Tensor, y: torch.Tensor | None = None):
        z = self.backbone(phrase_emb)
        out = self.head(z, y)
        out["phrase_out"] = z
        return out

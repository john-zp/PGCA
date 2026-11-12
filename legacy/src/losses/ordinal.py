from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        logits = self.fc(x)
        out = {"logits": logits}
        if y is not None:
            y = y.view(-1).clamp(1, self.num_classes)
            targets = torch.zeros((y.size(0), self.num_classes - 1), dtype=torch.float32, device=x.device)
            for k in range(self.num_classes - 1):
                targets[:, k] = (y >= (k + 2)).float()
            out["loss_ord"] = F.binary_cross_entropy_with_logits(logits, targets)
        p_ge = torch.sigmoid(logits)
        pred = 1 + (p_ge >= 0.5).sum(dim=1)
        out["p_ge"] = p_ge
        out["pred"] = pred
        return out

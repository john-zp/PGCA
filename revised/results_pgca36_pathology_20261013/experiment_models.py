"""Reuse the frozen PGCA topology; never load historical checkpoints."""
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "audit_20261013/reference_B/src"))
from models.image_encoder import ResNetTokenEncoder
from models.pgca_model import PGCAModel


class IndependentOrdinalHead(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, 5)

    def forward(self, x, y=None):
        logits = self.fc(x)
        probability = logits.sigmoid()
        out = {"logits": logits, "p_ge": probability, "pred": 1 + (probability >= .5).sum(dim=1)}
        if y is not None:
            if y.dtype != torch.long or y.ndim != 1 or not ((y >= 1) & (y <= 6)).all():
                raise ValueError("Labels must be integer-coded 1..6, never clipped")
            targets = (y[:, None] >= torch.arange(2, 7, device=y.device)[None]).to(logits.dtype)
            out["loss_ord"] = F.binary_cross_entropy_with_logits(logits, targets)
        return out


MODALITIES = {"F": "F", "I": "I", "P": "P", "I+F": "IF", "I+P": "IP", "F+P": "FP", "I+F+P-simple": "IFP"}


class SimpleFusion(nn.Module):
    def __init__(self, modalities):
        super().__init__()
        self.modalities = modalities
        if "I" in modalities:
            self.image_encoder = ResNetTokenEncoder(1, 256)
        if "F" in modalities:
            self.feature_encoder = nn.Sequential(nn.Linear(36, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, 256))
        if "P" in modalities:
            self.prompt_proj = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256))
        self.fusion = nn.Sequential(nn.Linear(256 * len(modalities), 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, 256), nn.ReLU())
        self.head = IndependentOrdinalHead()

    def forward(self, image, features, phrase_emb, y=None):
        parts = []
        if "I" in self.modalities:
            _, global_image, _ = self.image_encoder(image)
            parts.append(global_image)
        if "P" in self.modalities:
            parts.append(self.prompt_proj(phrase_emb))
        if "F" in self.modalities:
            parts.append(self.feature_encoder(features))
        return self.head(self.fusion(torch.cat(parts, dim=1)), y)


def make_model(name):
    if name in MODALITIES:
        return SimpleFusion(MODALITIES[name])
    if name.startswith("PGCA-"):
        model = PGCAModel(512, 36, 256, 256, 4, .2, 6)
        model.head = IndependentOrdinalHead()
        return model
    raise ValueError("Unknown requested model: " + name)

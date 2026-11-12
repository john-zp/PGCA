from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead
from models.image_encoder import ResNetTokenEncoder
from models.pgca_model import PromptCrossAttention


class ImagePhraseModel(nn.Module):
    def __init__(self, phrase_dim: int, image_embed_dim: int, hidden_dim: int, attn_heads: int, dropout: float, num_classes: int):
        super().__init__()
        self.image_encoder = ResNetTokenEncoder(in_channels=1, embed_dim=image_embed_dim)
        self.prompt_proj = nn.Sequential(
            nn.Linear(phrase_dim, image_embed_dim),
            nn.GELU(),
            nn.LayerNorm(image_embed_dim),
        )
        self.cross_attn = PromptCrossAttention(image_embed_dim, num_heads=attn_heads, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(image_embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = CORNHead(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, phrase_emb: torch.Tensor, y: torch.Tensor | None = None):
        image_tokens, image_global, grid_hw = self.image_encoder(image)
        prompt = self.prompt_proj(phrase_emb)
        prompt_fused, attn_map = self.cross_attn(image_tokens, prompt)
        z = self.fusion(torch.cat([image_global, prompt_fused], dim=1))
        out = self.head(z, y)
        out["attn_map"] = attn_map
        out["grid_hw"] = grid_hw
        out["fusion_out"] = z
        return out
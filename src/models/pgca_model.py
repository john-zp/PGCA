from __future__ import annotations

import torch
import torch.nn as nn

from losses.ordinal import CORNHead
from models.image_encoder import ResNetTokenEncoder


class PromptCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens: torch.Tensor, prompt_emb: torch.Tensor):
        query = self.q_proj(prompt_emb).unsqueeze(1)
        key = self.k_proj(image_tokens)
        value = self.v_proj(image_tokens)
        fused, attn = self.mha(query, key, value, need_weights=True, average_attn_weights=False)
        attn_map = attn.mean(dim=1)  # [B,1,N]
        x = self.ln1(query + fused)
        x = self.ln2(x + self.ffn(x))
        return x.squeeze(1), attn_map


class PGCAModel(nn.Module):
    def __init__(
        self,
        phrase_dim: int,
        feature_dim: int,
        image_embed_dim: int,
        hidden_dim: int,
        attn_heads: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.image_encoder = ResNetTokenEncoder(in_channels=1, embed_dim=image_embed_dim)
        self.prompt_proj = nn.Sequential(
            nn.Linear(phrase_dim, image_embed_dim),
            nn.GELU(),
            nn.LayerNorm(image_embed_dim),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_embed_dim),
        )
        self.cross_attn = PromptCrossAttention(image_embed_dim, num_heads=attn_heads, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(image_embed_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = CORNHead(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, features: torch.Tensor, phrase_emb: torch.Tensor, y: torch.Tensor | None = None):
        image_tokens, image_global, grid_hw = self.image_encoder(image)
        prompt = self.prompt_proj(phrase_emb)
        prompt_fused, attn_map = self.cross_attn(image_tokens, prompt)
        feature_vec = self.feature_encoder(features)
        fused = self.fusion(torch.cat([image_global, prompt_fused, feature_vec], dim=1))
        out = self.head(fused, y)
        out["attn_map"] = attn_map
        out["grid_hw"] = grid_hw
        out["fusion_out"] = fused
        return out

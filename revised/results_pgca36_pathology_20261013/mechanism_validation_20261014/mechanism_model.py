"""Strict PGCA-0 ablation: remove content-dependent QK weights only."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
from experiment_models import make_model
from models.pgca_model import PromptCrossAttention


class UniformCrossAttention(PromptCrossAttention):
    def forward(self, image_tokens, prompt_emb):
        query = self.q_proj(prompt_emb).unsqueeze(1)
        key = self.k_proj(image_tokens)
        value = self.v_proj(image_tokens)
        # Identical keys at every position make softmax exactly uniform before
        # the original attention dropout. Keep query in the original residual.
        fused, attn = self.mha(query * 0, key * 0, value,
                               need_weights=True, average_attn_weights=False)
        x = self.ln1(query + fused)
        x = self.ln2(x + self.ffn(x))
        return x.squeeze(1), attn.mean(dim=1)


def make_strict_ablation():
    model = make_model('PGCA-0')
    # No new parameters, reinitialization, module removal, or RNG consumption.
    model.cross_attn.__class__ = UniformCrossAttention
    return model

"""Separate attention-selection semantics from the existing text residual."""
import os
import sys
from pathlib import Path
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("PGCA_ROOT", Path(__file__).resolve().parents[1]))
BASE = ROOT / "results_pgca36_pathology_20261013"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
from experiment_models import make_model
from train_one import Cohort


def isolated_forward(self, image, features, phrase_emb, y=None, query_emb=None):
    tokens, image_global, grid = self.image_encoder(image)
    prompt = self.prompt_proj(phrase_emb)
    ca = self.cross_attn
    residual = ca.q_proj(prompt).unsqueeze(1)
    query = residual if query_emb is None else ca.q_proj(self.prompt_proj(query_emb)).unsqueeze(1)
    value, weights = ca.mha(query, ca.k_proj(tokens), ca.v_proj(tokens),
                            need_weights=True, average_attn_weights=False)
    x = ca.ln1(residual + value)
    x = ca.ln2(x + ca.ffn(x)).squeeze(1)
    feature = self.feature_encoder(features)
    fusion = self.fusion(torch.cat([image_global, x, feature], dim=1))
    out = self.head(fusion, y)
    out.update(attn_map=weights.mean(1), grid_hw=grid, fusion_out=fusion)
    return out


def query_model():
    model = make_model("PGCA-0")
    model.forward = MethodType(isolated_forward, model)
    return model


class QueryCohort(Cohort):
    def __init__(self, split, seed, condition):
        super().__init__(split, seed, shuffled=False)
        self.condition = condition
        self.query_donors = np.load(HERE / "design/query_donors.npy")
        self.constant = np.load(HERE / "design/train_mean_embedding.npy")

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if self.condition == "Shuffled":
            vector = self.arrays["pathology_embeddings"][self.query_donors[data["index"]]]
        elif self.condition == "Constant":
            vector = self.constant
        else:
            return data
        data["query_emb"] = torch.from_numpy(np.array(vector, dtype=np.float32, copy=True))
        return data


def inputs(data):
    return {key: data[key].cuda() for key in ["image", "features", "phrase_emb", "y", "query_emb"] if key in data}

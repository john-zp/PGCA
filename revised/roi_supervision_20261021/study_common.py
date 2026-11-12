"""Read-only integration with the frozen PGCA experiment."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results_pgca36_pathology_20261013"
SUP = ROOT / "supplementary_validation_20261014"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import json
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_one import Cohort, evaluate, inputs, save_json, save_checkpoint, set_seed, sha
from experiment_models import make_model
from spatial_losses import alignment, spatial_distributions

SEEDS = [42, 3407, 2026]
FAMILIES = ["PGCA-ROI-KL", "PGCA-staticOT-ROI-KL"]
WEIGHTS = [.01, .05, .2]
HISTORICAL = ["PGCA-0", "PGCA-staticOT", "PGCA-full-SBpath"]


def read_json(path):
    return json.loads(Path(path).read_text())


def git_commit(path=HERE):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def config_for(family, seed, weight):
    config = read_json(BASE / "common_config.json")
    config.update(model=family, seed=seed, lambda_ROI=weight, lambda_SB_path=0,
                  lambda_static_OT=.2 if family==FAMILIES[1] else 0.,
                  code_commit=git_commit(), historical_code_commit=git_commit(BASE),
                  protocol_sha256=sha(HERE/"protocol.md"), roi_log_floor=1e-8,
                  ROI_KL_reduction="sum_tokens_mean_valid_cases", test_hyperparameter_selection=False)
    return config


def roi_kl(attention, roi, grid):
    alpha, mu, valid = spatial_distributions(attention, roi, grid)
    if alpha is None:
        return attention.sum()*0, {"valid_roi":0,"alpha_mass_error":0.,"mu_mass_error":0.,"floored_entries":0}
    log_alpha = alpha.clamp_min(1e-8).log()
    log_alpha = log_alpha-torch.logsumexp(log_alpha,dim=-1,keepdim=True)
    loss = torch.nn.functional.kl_div(log_alpha,mu,reduction="batchmean")
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite ROI KL")
    return loss, {"valid_roi":int(valid.sum()),
                  "alpha_mass_error":float((alpha.sum(-1)-1).abs().max().detach()),
                  "mu_mass_error":float((mu.sum(-1)-1).abs().max().detach()),
                  "floored_entries":int((alpha<1e-8).sum())}


def objective(out, roi, config):
    kl, details = roi_kl(out["attn_map"],roi,out["grid_hw"])
    static = out["attn_map"].sum()*0
    ot_details={"marginal_error":0.,"endpoint_error":0.}
    if config["lambda_static_OT"]:
        static,_,ot_details=alignment(out["attn_map"],roi,out["grid_hw"],False,
            config["OT_epsilon"],config["sinkhorn_iterations"],config["static_OT_internal_beta"])
    total=out["loss_ord"]+config["lambda_ROI"]*kl+config["lambda_static_OT"]*static
    return total, {"ordinal":out["loss_ord"],"roi_kl":kl,"static_OT":static}, {**ot_details,**details}


def norm_of(tensors):
    values=[t.detach().double().square().sum() for t in tensors if t is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def attention_parameters(model):
    return [(name,p) for name,p in model.named_parameters() if "attn" in name or "attention" in name]


def verify_lock():
    lock=read_json(HERE/"source_lock.json")
    for path,expected in lock["read_only_hashes"].items():
        if sha(Path(path))!=expected:
            raise RuntimeError("Read-only source changed: "+path)
    for relative,expected in lock["training_code_hashes"].items():
        if sha(HERE/relative)!=expected:
            raise RuntimeError("New training source changed after preflight: "+relative)
    return lock

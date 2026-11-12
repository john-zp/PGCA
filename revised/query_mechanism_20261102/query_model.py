"""Controlled Query source and residual gates around the frozen PGCA graph."""
import os
from pathlib import Path
import sys
from types import MethodType

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results_pgca36_pathology_20261013'
ROI = ROOT/'roi_supervision_20261021'
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(ROI))

import numpy as np
import torch
from torch import nn
from experiment_models import make_model
from train_one import Cohort, set_seed, save_json, save_checkpoint, sha
from study_common import roi_kl

QUERIES = ['Matched','Shuffled','Learnable','Sonographic','Constant']


def controlled_forward(self, image, features, phrase_emb, y=None, query_emb=None):
    tokens, image_global, grid = self.image_encoder(image)
    ca = self.cross_attn
    matched_q = None
    if self.residual_on or self.query_source == 'Matched':
        matched_q = ca.q_proj(self.prompt_proj(phrase_emb)).unsqueeze(1)
    if self.query_source == 'Matched':
        query = matched_q
    elif self.query_source == 'Shuffled':
        if query_emb is None:
            raise ValueError('Shuffled requires the locked donor embedding')
        query = ca.q_proj(self.prompt_proj(query_emb)).unsqueeze(1)
    elif self.query_source == 'Learnable':
        query = ca.q_proj(self.learnable_query).view(1,1,256).expand(len(image),-1,-1)
    elif self.query_source == 'Sonographic':
        query = ca.q_proj(self.observation_query(features)).unsqueeze(1)
    elif self.query_source == 'Constant':
        query = self.frozen_query.view(1,1,256).expand(len(image),-1,-1)
    else:
        raise ValueError(self.query_source)
    residual = matched_q if self.residual_on else torch.zeros_like(query)
    value, weights = ca.mha(query,ca.k_proj(tokens),ca.v_proj(tokens),need_weights=True,average_attn_weights=False)
    x = ca.ln1(residual + value)
    x = ca.ln2(x + ca.ffn(x)).squeeze(1)
    feature = self.feature_encoder(features)
    fusion = self.fusion(torch.cat([image_global,x,feature],dim=1))
    out = self.head(fusion,y)
    out.update(attn_map=weights.mean(1),grid_hw=grid,fusion_out=fusion)
    if self.inspect_paths:
        out['pathways']={'query':query,'residual':residual,'image_global':image_global,'structured':feature,'attended_value':value}
    return out


def make_query_model(query, residual_on, seed):
    if query not in QUERIES:
        raise ValueError(query)
    model=make_model('PGCA-0')
    model.query_source,model.residual_on,model.inspect_paths=query,bool(residual_on),False
    # Isolate adapter initialization from the shared model and training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+61201102)
        if query=='Learnable':
            model.learnable_query=nn.Parameter(torch.empty(256).normal_(mean=0.,std=.02))
        elif query=='Sonographic':
            model.observation_query=nn.Sequential(nn.Linear(36,256),nn.GELU(),nn.LayerNorm(256))
    if query=='Constant':
        mean=torch.from_numpy(np.load(HERE/'design/train_mean_embedding.npy'))
        with torch.no_grad():
            fixed=model.cross_attn.q_proj(model.prompt_proj(mean)).detach().clone()
        model.register_buffer('frozen_query',fixed)
    if not residual_on and query in ['Learnable','Sonographic','Constant']:
        model.prompt_proj.requires_grad_(False)
    if not residual_on and query=='Constant':
        model.cross_attn.q_proj.requires_grad_(False)
    model.forward=MethodType(controlled_forward,model)
    return model


class QueryCohort(Cohort):
    def __init__(self,split,seed,query):
        super().__init__(split,seed,shuffled=False)
        self.query_source=query
        self.query_donors=np.load(HERE/'design/query_donors.npy')

    def __getitem__(self,item):
        data=super().__getitem__(item)
        if self.query_source=='Shuffled':
            donor=self.query_donors[data['index']]
            data['query_emb']=torch.from_numpy(np.array(self.arrays['pathology_embeddings'][donor],dtype=np.float32,copy=True))
        return data


def inputs(data,device='cuda',with_label=True):
    keys=['image','features','phrase_emb','query_emb']+(['y'] if with_label else [])
    return {key:data[key].to(device) for key in keys if key in data}


def objective(out,roi,weight):
    if weight:
        kl,details=roi_kl(out['attn_map'],roi,out['grid_hw'])
    else:
        kl=out['attn_map'].sum()*0
        details={'valid_roi':0,'alpha_mass_error':0.,'mu_mass_error':0.,'floored_entries':0}
    return out['loss_ord']+weight*kl,kl,details


def parameter_counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'added_query_adapter':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('learnable_query','observation_query')))}

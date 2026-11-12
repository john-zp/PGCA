"""Synthetic CPU contract checks only; no optimizer or clinical data."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1] / 'revised'
BASE = ROOT / 'results_pgca36_pathology_20261013'
sys.path[:0] = [str(ROOT / 'query_mechanism_20261102'), str(BASE),
               str(BASE / 'mechanism_validation_20261014')]

import numpy as np
import torch
import query_model as query
from evaluation import metrics
from experiment_models import IndependentOrdinalHead, make_model
from mechanism_model import make_strict_ablation
from spatial_losses import alignment, spatial_distributions
from study_common import roi_kl


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    image = torch.rand(2, 1, 224, 224)
    features = torch.randint(0, 2, (2, 36)).float()
    text = torch.randn(2, 512)
    donor = torch.randn(2, 512)
    y = torch.tensor([2, 5])
    torch.manual_seed(42)
    base = make_model('PGCA-0').eval()
    assert isinstance(base.head, IndependentOrdinalHead)
    assert sum(p.numel() for p in base.parameters()) == 2828197
    with torch.no_grad():
        original = base(image, features, text, y)
    assert original['logits'].shape == (2, 5)
    assert original['attn_map'].shape == (2, 1, 196)
    assert original['grid_hw'] == (14, 14)

    for residual in (True, False):
        for source in query.QUERIES:
            torch.manual_seed(42)
            # Only Constant needs a train-only mean; substitute a synthetic vector.
            with patch.object(query.np, 'load', return_value=np.zeros(512, dtype=np.float32)):
                model = query.make_query_model(source, residual, 42).eval()
            model.inspect_paths = True
            out = model(image, features, text, y, query_emb=donor)
            assert torch.isfinite(out['loss_ord']) and torch.isfinite(out['attn_map']).all()
            torch.testing.assert_close(out['attn_map'].sum(-1), torch.ones(2, 1))
            if residual:
                expected = model.cross_attn.q_proj(model.prompt_proj(text)).unsqueeze(1)
                torch.testing.assert_close(out['pathways']['residual'], expected)
            else:
                assert not out['pathways']['residual'].count_nonzero()
            if source == 'Matched' and residual:
                torch.testing.assert_close(out['logits'], original['logits'], rtol=0, atol=0)
                torch.testing.assert_close(out['attn_map'], original['attn_map'], rtol=0, atol=0)
            if not residual and source in ('Learnable', 'Sonographic', 'Constant'):
                changed = model(image, features, donor, y, query_emb=donor)
                torch.testing.assert_close(changed['logits'], out['logits'], rtol=0, atol=0)
                torch.testing.assert_close(changed['attn_map'], out['attn_map'], rtol=0, atol=0)
            if source == 'Shuffled':
                changed = model(image, features, text, y, query_emb=text)
                torch.testing.assert_close(changed['pathways']['residual'], out['pathways']['residual'])
                assert not torch.equal(changed['pathways']['query'], out['pathways']['query'])
            out['loss_ord'].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cross_attn.parameters())
    uniform = make_strict_ablation().eval()
    assert sum(p.numel() for p in uniform.parameters()) == 2828197
    with torch.no_grad():
        out = uniform(image, features, text)
        torch.testing.assert_close(out['attn_map'], torch.full((2, 1, 196), 1 / 196))

    raw = torch.randn(2, 1, 16, requires_grad=True)
    attention = raw.softmax(-1)
    roi = torch.zeros(2, 1, 16, 16)
    roi[0, :, :8, :8] = 1
    alpha, mu, valid = spatial_distributions(attention, roi, (4, 4))
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(alpha.sum(-1), torch.ones(1))
    torch.testing.assert_close(mu.sum(-1), torch.ones(1))
    kl, _ = roi_kl(attention, roi, (4, 4))
    static, path, details = alignment(attention, roi, (4, 4), with_path=True)
    assert details['endpoint_error'] <= 1e-4 and details['marginal_error'] <= 1e-4
    (kl + static + path).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0].abs().sum() > 0
    empty, _ = roi_kl(attention, torch.zeros_like(roi), (4, 4))
    assert empty.item() == 0
    values = metrics([1, 2, 3, 4, 5, 6], [1, 3, 3, 4, 5, 6])
    assert values['MAE'] + 1e-12 >= 1 - values['Accuracy']
    print('PASS: CPU synthetic graphs, residual isolation, Uniform-QK, spatial gradients and metric contract')
    print('No optimizer, checkpoint, clinical data, GPU training or clinical inference used.')


if __name__ == '__main__':
    main()

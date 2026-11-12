"""Only three newly authorized strict mechanism-off runs, on the frozen cohort."""
import argparse
import json
import time
from pathlib import Path

from mechanism_model import BASE, make_model, make_strict_ablation
import torch
import numpy as np
from torch.utils.data import DataLoader
from train_one import (Cohort, inputs, evaluate, set_seed, save_json,
                       save_checkpoint, sha)

HERE = Path(__file__).resolve().parent
NAME = 'PGCA-0-uniform-QK'


def finite_backward(loss, model):
    assert torch.isfinite(loss).all(), 'Nonfinite loss'
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None), 'Nonfinite gradient'


def preflight(config):
    set_seed(42)
    original = make_model('PGCA-0').cuda().eval()
    set_seed(42)
    model = make_strict_ablation().cuda().eval()
    assert list(original.state_dict()) == list(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    train = Cohort('train', 42)
    data = next(iter(DataLoader(train, batch_size=4)))
    x = inputs(data)
    out = model(**x)
    assert out['logits'].shape == (4, 5) and out['fusion_out'].shape == (4, 256)
    assert out['grid_hw'] == (14, 14)
    assert torch.allclose(out['attn_map'], torch.full_like(out['attn_map'], 1 / 196), atol=1e-8, rtol=0)
    tokens, _, _ = model.image_encoder(x['image'])
    prompt = model.prompt_proj(x['phrase_emb'])
    ca = model.cross_attn
    with torch.no_grad():
        value = ca.v_proj(tokens)
        d = ca.mha.embed_dim
        projected_value = torch.nn.functional.linear(value, ca.mha.in_proj_weight[2*d:], ca.mha.in_proj_bias[2*d:])
        pooled = ca.mha.out_proj(projected_value.mean(1, keepdim=True))
        manual = ca.ln1(ca.q_proj(prompt).unsqueeze(1) + pooled)
        manual = ca.ln2(manual + ca.ffn(manual)).squeeze(1)
        actual, weights = ca(tokens, prompt)
        assert torch.allclose(manual, actual, atol=2e-6, rtol=1e-5)
        _, altered = ca(tokens.flip(1) * 3, prompt.flip(0) * 4)
        assert torch.equal(weights, altered), 'Weights still depend on text/image content'
    finite_backward(out['loss_ord'], model)
    grad = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in model.named_parameters()}
    for prefix in ['image_encoder.stage1', 'feature_encoder.0', 'prompt_proj.0', 'cross_attn.q_proj', 'cross_attn.v_proj']:
        assert any(v is not None and v > 0 for k, v in grad.items() if k.startswith(prefix)), prefix
    assert grad['cross_attn.k_proj.weight'] == 0
    assert model.cross_attn.mha.in_proj_weight.grad[:512].abs().max() == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    smoke = []
    model.train()
    for i, batch in enumerate(DataLoader(train, batch_size=4)):
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs(batch))
        finite_backward(output['loss_ord'], model)
        optimizer.step()
        smoke.append(float(output['loss_ord'].detach()))
        if i == 1:
            break
    del model, original, optimizer
    torch.cuda.empty_cache()
    report = {'status': 'PASS', 'identical_initial_state_and_parameter_shapes': True,
              'uniform_attention_content_independence': True, 'manual_uniform_value_pooling_equivalence': True,
              'gradient_maxima': grad, 'two_batch_losses': smoke,
              'parameter_note': 'External q_proj remains active through the text residual; external k_proj and internal MHA Q/K receive zero gradients.'}
    save_json(HERE / 'preflight_checks.json', report)
    return report


def train(seed, config, lock):
    run = HERE / 'runs' / NAME / f'seed_{seed}'
    if (run / 'completed.json').exists():
        return
    if run.exists():
        raise RuntimeError('Refuse to overwrite an incomplete run: ' + str(run))
    run.mkdir(parents=True)
    config = dict(config, model=NAME, seed=seed, code_commit=lock['code_commit'],
                  base_code_commit=lock['base_source_lock']['code_commit'],
                  dataset_hashes=lock['base_source_lock']['dataset_hashes'],
                  effective_lambda_static_OT=0, effective_lambda_SB_path=0,
                  ablation='Uniform spatial weights; original text residual, V/output projections, FFN and fusion retained')
    save_json(run / 'config.json', config)
    save_json(run / 'environment.json', lock)
    (run / 'split_manifest.csv').write_bytes((BASE / 'split_manifest.csv').read_bytes())
    set_seed(seed)
    model = make_strict_ablation().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    trainset, val, test = [Cohort(s, seed) for s in ['train', 'val', 'test']]
    best, best_epoch, curve = (-float('inf'), -float('inf')), 0, []
    for epoch in range(1, config['epochs'] + 1):
        set_seed(seed + epoch)
        trainset.epoch = epoch
        model.train()
        total, examples = 0., 0
        start = time.monotonic()
        generator = torch.Generator().manual_seed(seed + epoch)
        for data in DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs(data))
            finite_backward(out['loss_ord'], model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            optimizer.step()
            total += float(out['loss_ord'].detach()) * len(data['y'])
            examples += len(data['y'])
        validation = evaluate(model, val, None, 'val')
        selection = (validation['QWK'], -validation['MAE'])
        if selection > best:
            best, best_epoch = selection, epoch
            save_checkpoint(run / 'best_checkpoint.pt', {'model': model.state_dict(), 'epoch': epoch, 'config': config, 'validation': validation})
            save_json(run / 'best_validation_metrics.json', validation)
        curve.append({'epoch': epoch, 'training_loss': total / examples, 'validation': validation, 'elapsed_seconds': time.monotonic() - start})
        save_json(run / 'training_curve.json', curve)
        save_checkpoint(run / 'last_checkpoint.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'selection': best, 'best_epoch': best_epoch, 'config': config, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(), 'python_rng': __import__('random').getstate()})
        print(json.dumps({'seed': seed, **{k: curve[-1][k] for k in ['epoch', 'training_loss', 'elapsed_seconds']}, 'val_QWK': validation['QWK']}), flush=True)
    checkpoint = torch.load(run / 'best_checkpoint.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result = evaluate(model, test, run, 'test', with_maps=False)
    save_json(run / 'completed.json', {'model': NAME, 'seed': seed, 'best_epoch': best_epoch, 'test': result,
                                     'code_commit': lock['code_commit'], 'text_role': 'pathology known at training and inference'})
    print('COMPLETED ' + str(run), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--seed', type=int, choices=[42, 3407, 2026])
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    config = json.loads((BASE / 'common_config.json').read_text())
    lock = json.loads((HERE / 'source_lock.json').read_text())
    for relative, expected in lock['code_hashes'].items():
        assert sha(HERE / relative) == expected, relative
    if args.preflight:
        preflight(config)
    else:
        assert json.loads((HERE / 'preflight_checks.json').read_text())['status'] == 'PASS'
        train(args.seed, config, lock)


if __name__ == '__main__':
    main()

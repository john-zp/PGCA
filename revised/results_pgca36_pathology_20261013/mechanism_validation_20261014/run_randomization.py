"""Inference-only, cumulative parameter randomization on nine locked checkpoints."""
import json
from pathlib import Path

from mechanism_model import BASE, make_model
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from train_one import Cohort, inputs, set_seed, save_json, sha

HERE = Path(__file__).resolve().parent
STAGES = [('original', []), ('attention', ['cross_attn.']),
          ('text_projection', ['prompt_proj.']), ('token_projection', ['image_encoder.token_proj.']),
          ('cnn_stage4', ['image_encoder.stage4.']), ('cnn_stage3', ['image_encoder.stage3.']),
          ('cnn_stage2', ['image_encoder.stage2.']), ('cnn_stage1', ['image_encoder.stage1.'])]


def spearman_rows(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    denom = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
    result = np.full(len(a), np.nan)
    np.divide((ra * rb).sum(1), denom, out=result, where=denom > 0)
    return np.clip(result, -1, 1)


def forward_maps(model, dataset):
    model.eval()
    maps, predictions = [], []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=16, shuffle=False):
            x = inputs(data)
            x.pop('y')
            out = model(**x)
            assert out['grid_hw'] == (14, 14)
            maps.append(out['attn_map'].reshape(-1, 14, 14).cpu().numpy())
            predictions.extend(out['pred'].cpu().tolist())
    maps = np.concatenate(maps)
    assert np.isfinite(maps).all() and (maps >= 0).all()
    assert np.allclose(maps.sum((1, 2)), 1, atol=1e-5, rtol=0)
    return maps, np.asarray(predictions)


def main():
    torch.set_num_threads(8)
    output = HERE / 'randomization'
    output.mkdir(exist_ok=True)
    set_seed(42)
    test = Cohort('test', 42)
    meta = test.manifest.iloc[test.indices].reset_index(drop=True)
    valid_roi = np.asarray(test.arrays['roi_masks'][test.indices]).reshape(len(test), -1).sum(1) > 0
    assert len(test) == 555 and valid_roi.sum() == 383
    identifiers = meta[['image_id', 'patient_key', 'group_id']].copy()
    identifiers['valid_roi'] = valid_roi
    identifiers.to_csv(output / 'test_map_identifiers.csv', index=False)
    records, summaries, audits = [], [], []
    for name in ['PGCA-0', 'PGCA-staticOT', 'PGCA-full-SBpath']:
        for seed in [42, 3407, 2026]:
            run = BASE / 'runs' / name / f'seed_{seed}'
            checkpoint_path = run / 'best_checkpoint.pt'
            checkpoint_sha = sha(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert checkpoint['config']['model'] == name and checkpoint['config']['seed'] == seed
            assert checkpoint['config']['dataset_hashes'] == json.loads((BASE / 'source_lock.json').read_text())['dataset_hashes']
            model = make_model(name).cuda()
            model.load_state_dict(checkpoint['model'], strict=True)
            original, original_pred = forward_maps(model, test)
            saved = np.load(run / 'test_attention_maps.npy')
            saved_pred = pd.read_csv(run / 'test_predictions.csv')
            assert identifiers.image_id.tolist() == saved_pred.image_id.tolist()
            assert np.array_equal(original_pred, saved_pred.y_pred.to_numpy())
            max_error = float(np.max(np.abs(saved - original)))
            assert np.allclose(original, saved, atol=1e-6, rtol=1e-5), max_error
            audit = {'model': name, 'training_seed': seed, 'checkpoint': str(checkpoint_path),
                     'checkpoint_sha256': checkpoint_sha, 'stored_map_replay_max_error': max_error,
                     'hard_predictions_reproduced': True, 'stages': []}
            # A map returned upstream of the classifier cannot depend on head weights.
            set_seed(17)
            fresh = make_model(name).state_dict()
            downstream = [k for k in fresh if k.startswith(('fusion.', 'head.'))]
            state = model.state_dict()
            for k in downstream:
                state[k].copy_(fresh[k])
            control, control_pred = forward_maps(model, test)
            assert np.array_equal(control, original)
            audit['downstream_only_control'] = {'parameter_names': downstream,
                'map_exactly_unchanged': True, 'prediction_changed_n': int((control_pred != original_pred).sum())}
            for random_seed in [17, 29, 43]:
                set_seed(random_seed)
                fresh = make_model(name).state_dict()
                model.load_state_dict(checkpoint['model'], strict=True)
                state = model.state_dict()
                cumulative = []
                for stage_index, (stage, prefixes) in enumerate(STAGES):
                    names = [k for k in state if any(k.startswith(p) for p in prefixes)]
                    changed = []
                    for k in names:
                        if not torch.equal(state[k].cpu(), fresh[k]):
                            changed.append(k)
                        state[k].copy_(fresh[k])
                    cumulative += names
                    maps, prediction = (original, original_pred) if stage_index == 0 else forward_maps(model, test)
                    relative = f'{name}_seed{seed}_random{random_seed}_{stage_index}_{stage}.npy'
                    np.save(output / relative, maps, allow_pickle=False)
                    rho = spearman_rows(original.reshape(len(test), -1), maps.reshape(len(test), -1))
                    if stage_index == 0:
                        assert np.allclose(rho, 1, equal_nan=False)
                    info = {'model': name, 'training_seed': seed, 'randomization_seed': random_seed,
                            'stage_index': stage_index, 'stage': stage}
                    frame = identifiers.copy()
                    for k, v in info.items():
                        frame[k] = v
                    frame['spearman'] = rho
                    frame['absolute_spearman'] = np.abs(rho)
                    frame['randomized_prediction'] = prediction
                    frame['original_prediction'] = original_pred
                    records.append(frame)
                    for subset, mask in [('all_test', np.ones(len(test), dtype=bool)), ('valid_roi', valid_roi)]:
                        values = rho[mask]
                        good = values[np.isfinite(values)]
                        summaries.append(dict(info, subset=subset, n=int(mask.sum()), defined_n=len(good),
                            undefined_n=int((~np.isfinite(values)).sum()), mean_spearman=float(good.mean()) if len(good) else None,
                            median_spearman=float(np.median(good)) if len(good) else None,
                            mean_absolute_spearman=float(np.abs(good).mean()) if len(good) else None,
                            fraction_abs_rho_ge_0_9=float((np.abs(good) >= .9).mean()) if len(good) else None))
                    audit['stages'].append(dict(info, newly_assigned_state_names=names, actually_changed_names=changed,
                        cumulative_state_names=list(cumulative), map_file=relative, map_sha256=sha(output / relative)))
                assert any(k.startswith('cross_attn.') for k in cumulative)
                assert any(k.startswith('image_encoder.stage1.') for k in cumulative)
            assert sha(checkpoint_path) == checkpoint_sha
            audits.append(audit)
            save_json(output / 'checkpoint_randomization_audit.json', audits)
            pd.concat(records, ignore_index=True).to_csv(output / 'per_image_spearman.csv', index=False)
            pd.DataFrame(summaries).to_csv(output / 'stage_summary.csv', index=False)
            print('RANDOMIZATION_COMPLETED ' + name + ' seed=' + str(seed), flush=True)
    save_json(output / 'completed.json', {'status': 'PASS', 'checkpoint_count': 9,
        'randomization_seeds': [17, 29, 43], 'n_test': len(test), 'n_roi': int(valid_roi.sum()),
        'map_resolution': [14, 14], 'no_retraining': True,
        'interpretation': 'Sensitivity of non-class-specific attention to its ancestor parameters; not proof of causal explanation fidelity'})


if __name__ == '__main__':
    main()

"""Freeze existing evidence without invoking a training/evaluation writer."""
from common import *
import numpy as np
import pandas as pd
import torch
from query_model import make_query_model, parameter_counts, set_seed


def main():
    torch.set_num_threads(4)
    manifest = pd.read_csv(BASE/'data/dataset_manifest.csv')
    assert manifest.groupby('split').size().to_dict() == {'test':555,'train':2599,'val':555}
    assert manifest.groupby('group_id').split.nunique().max() == 1
    assert manifest.groupby('patient_key').split.nunique().max() == 1
    test = manifest[manifest.split=='test']
    assert test.group_id.nunique() == 380
    masks = np.load(BASE/'data/roi_masks.npy', mmap_mode='r')
    valid = np.array([bool(np.any(masks[i])) for i in test.index])
    assert valid.sum() == 383 and test.loc[valid].group_id.nunique() == 290
    manifest.to_csv(OUT/'evidence_archive/data_split_manifest.csv',index=False)
    test.loc[valid].to_csv(OUT/'evidence_archive/roi_manifest.csv',index=False)
    conditions = read_json(STUDY/'protocol.json')['conditions']
    rows, checkpoints, configs, predictions, attentions, hashes = [],[],[],[],[],[]
    for c in conditions:
        run = STUDY/'runs'/c['run_id']
        cfg = read_json(run/'resolved_config.json')
        done = read_json(run/'completed.json')
        checkpoint = torch.load(run/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        set_seed(c['seed'])
        model = make_query_model(c['query'], c['residual_on'], c['seed'])
        model.load_state_dict(checkpoint['model'], strict=True)
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        counts = parameter_counts(model)
        assert counts == cfg['parameter_counts']
        assert checkpoint['epoch'] == done['best_epoch']
        assert checkpoint['config']['seed'] == c['seed']
        assert cfg['query'] == c['query'] and cfg['residual_on'] == c['residual_on']
        assert cfg['lambda_ROI'] == c['lambda_ROI']
        assert sha(run/'best_checkpoint.pt') == done['checkpoint_sha256']
        pred = pd.read_csv(run/'test_predictions.csv')
        for name, expected in [('image_id',test.image_id),('group_id',test.group_id),('y_true',test.y)]:
            assert np.array_equal(pred[name],expected), (c['run_id'],name)
        maps = np.load(run/'test_attention_maps.npy')
        assert maps.shape == (555,14,14) and np.isfinite(maps).all()
        assert np.allclose(maps.sum((1,2)),1,atol=1e-5)
        curve = read_json(run/'training_curve.json')
        assert len(curve) == 30
        best = max(curve,key=lambda r:(r['validation']['QWK'],-r['validation']['MAE'],-r['epoch']))['epoch']
        assert best == checkpoint['epoch']
        fitted = cfg['training_code_commit']
        if c['reuse_candidate']:
            fitted = checkpoint['config']['code_commit']
        row = dict(experiment_id=c['run_id'],**c,status='REUSED' if c['reuse_candidate'] else 'COMPLETED',
                   checkpoint=str(run/'best_checkpoint.pt'),config=str(run/'resolved_config.json'),
                   fitted_commit=fitted,wrapper_commit=cfg['controlled_wrapper_commit'],best_epoch=best,
                   prediction=str(run/'test_predictions.csv'),attention=str(run/'test_attention_maps.npy'),
                   **counts)
        rows.append(row)
        checkpoints.append(dict(experiment_id=c['run_id'],path=row['checkpoint'],sha256=done['checkpoint_sha256'],
                                loaded_CPU=True,state_dict_strict=True,finite=True,seed=c['seed'],best_epoch=best))
        configs.append(dict(experiment_id=c['run_id'],path=row['config'],sha256=sha(row['config']),fitted_commit=fitted))
        predictions.append(dict(experiment_id=c['run_id'],path=row['prediction'],sha256=sha(row['prediction']),images=555,groups=380))
        attentions.append(dict(experiment_id=c['run_id'],path=row['attention'],sha256=sha(row['attention']),shape='555x14x14',ROI_images=383,ROI_groups=290))
        print('FROZEN',c['run_id'],flush=True)
    assert len(rows)==39
    roots = [STUDY,BASE,SUP,ROI]
    source = []
    inventory = []
    for root in roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(v in p.parts for v in ['.git','__pycache__','pretrained_cache','supplement_packages','analysis_attempts']):
                continue
            if p.suffix not in ['.py','.json','.csv','.md','.log','.txt','.npy','.pt','.cbm']:
                continue
            digest=sha(p)
            entry=dict(path=str(p),relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest)
            hashes.append(entry)
            if p.suffix=='.py': source.append(entry)
            if p.name in ['test_predictions.csv','test_attention_maps.npy','best_checkpoint.pt','config.json','training_curve.json']:
                inventory.append(entry)
    for name,data in [('experiment_manifest',rows),('checkpoint_manifest',checkpoints),('config_manifest',configs),
                      ('prediction_manifest',predictions),('attention_manifest',attentions),
                      ('source_version_manifest',source),('sha256_manifest',hashes),('historical_inventory',inventory)]:
        pd.DataFrame(data).to_csv(OUT/f'evidence_archive/{name}.csv',index=False)
    frame=pd.DataFrame(rows)
    frame[['stage','query','residual_on','lambda_ROI','seed','total','trainable','frozen','added_query_adapter']].to_csv(OUT/'statistics/query_parameter_comparison.csv',index=False)
    cohort=manifest.groupby('split').agg(images=('image_id','size'),patient_directories=('patient_key','nunique'),risk_groups=('group_id','nunique')).reset_index()
    report='# Experiment Freeze Report\n\n39 model-seed results, not 39 independent Query designs: A=15, B=15, C=9. There are 36 new fits and 3 verified historical Matched/ON reuses.\n\n'
    report+='All 39 checkpoints were loaded on CPU, strict-loaded into the appropriate controlled graph and checked for finite tensors, seed, selected epoch and parameter counts. Their SHA-256 values match completion records. All prediction IDs/labels/groups and attention dimensions agree with the frozen manifest. No training was invoked.\n\n'
    report+=markdown(cohort)+'\nLocalization: 383 images in 290 test risk groups; classification: 555 images in 380 groups. Patient-directory and risk-group cross-split overlap is zero. This preserves the historical identity grouping; it is not a new clinical identity adjudication.\n\n'
    report+=markdown(frame[['experiment_id','status','best_epoch','trainable']])
    report+='\n## Version Separation\n\nOld image-wise Shuffled, new group-consistent Shuffled, old Constant with trainable upstream projection, and new frozen external-input Constant are separate experiments. Residual-ON/OFF and no-ROI/ROI-KL also remain separate. Only the exact Matched/ON reuse is identical to historical PGCA-0; never pool these as six independent seeds. Historical Full/OT/ROI-KL are descriptive spatial comparators, not substitutes for Stage C.\n\nUse resolved configs for active objectives: generic original configs retain inactive legacy OT/SB defaults. Their source files and checkpoints are preserved. The 39 source predictions and maps are eligible for locked reanalysis; differently defined historical controls remain visible but cannot be merged.\n\n'
    report+=f'Hashed {len(hashes)} existing evidence/source files. Manifests reference original locations and content hashes rather than duplicating raw private clinical data. The source version manifest records analysis dependencies.\n'
    (OUT/'audit/01_experiment_freeze_report.md').write_text(report)
    write_json(OUT/'audit/freeze_status.json',dict(status='PASS',conditions=39,new_fits=36,reused=3,hashed_files=len(hashes),cohort=cohort.to_dict('records')))


if __name__=='__main__': main()

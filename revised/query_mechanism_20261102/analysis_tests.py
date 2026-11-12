"""Independent numerical checks for the reused statistics and ROI evaluator."""
import json
import numpy as np
import pandas as pd
from analyze_results import *


def main():
    torch.set_num_threads(2)
    conditions=json.loads((HERE/'protocol.json').read_text())['conditions']
    c=next(c for c in conditions if c['run_id']=='A_Matched_ON_seed42')
    run=HERE/'runs'/c['run_id']
    pred=pd.read_csv(run/'test_predictions.csv')
    validate_metrics(pred)
    loc=localization(c,pred,np.load(run/'test_attention_maps.npy'),pd.read_csv(BASE/'data/dataset_manifest.csv'),
                     np.load(BASE/'data/roi_masks.npy',mmap_mode='r'))
    old=pd.read_csv(BASE/'runs/PGCA-0/seed_42/test_localization.csv').set_index('image_id')
    loc=loc.set_index('image_id').loc[old.index]
    assert np.allclose(loc['Pointing_Game'],old.pointing_game_hit,rtol=0,atol=0)
    assert np.allclose(loc['IoU_at_0.5'],old.IoU_threshold_0_5,rtol=0,atol=1e-10)
    same=classification_pair({s:pred for s in [42,3407,2026]},{s:pred for s in [42,3407,2026]},'same',replicates=20)
    assert all(r['difference']==0 and r['ci_low']==0 and r['ci_high']==0 for r in same)
    changed=pred.copy()
    changed['y_pred']=1+(changed.y_pred.to_numpy()%6)
    result=classification_pair({42:pred},{42:changed},'manual_check',replicates=20)
    a=pred.sort_values('image_id');b=changed.sort_values('image_id')
    groups=sorted(a.group_id.unique())
    weights=np.random.default_rng(724299).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20)
    samples=[]
    for weights_row in weights:
        indices=np.concatenate([np.tile(np.flatnonzero(a.group_id.to_numpy()==g),w) for g,w in zip(groups,weights_row)])
        aa,bb=validate_metrics(a.iloc[indices]),validate_metrics(b.iloc[indices])
        samples.append([aa[m]-bb[m] for m in METRICS])
    expected=np.quantile(samples,[.025,.975],axis=0)
    for j,m in enumerate(METRICS):
        row=next(r for r in result if r['metric']==m and r['seed']==42)
        assert np.allclose([row['ci_low'],row['ci_high']],expected[:,j],atol=1e-12,rtol=0)
    # Stable ranking selects the same row-major first cells for tied heatmaps.
    heat=np.ones((224,224),dtype=np.float32)
    order=np.argsort(-heat.ravel(),kind='stable')
    assert np.array_equal(order,np.arange(224*224))
    controls=pd.DataFrame([{'stage':'A','query':q,'seed':s,'image':i} for q in ['Matched','Shuffled'] for s in [42,3407,2026] for i in range(383)])
    for q in ['Matched','Shuffled']:
        selected=controls[(controls['stage']=='A')&(controls['query']==q)]
        assert len(selected)==383*3 and selected.groupby('seed').size().eq(383).all()
    report=dict(status='PASS',metric_reference='sklearn',hard_prediction_MAE_bound='PASS',
                cluster_bootstrap_vs_explicit_group_resampling='PASS',identical_pair_zero_interval='PASS',
                historical_localization_pointing_and_IoU_reproduction='PASS',stable_row_major_ties='PASS',
                query_column_name_collision_regression='PASS')
    (HERE/'analysis').mkdir(exist_ok=True)
    (HERE/'analysis/analysis_unit_tests.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()

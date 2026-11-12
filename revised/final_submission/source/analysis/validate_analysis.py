"""Independent calculation checks and read-only evidence preservation gate."""
from common import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score
from statistics_common import cm_metrics


def metric(y,p):
    return np.array([accuracy_score(y,p),mean_absolute_error(y,p),cohen_kappa_score(y,p,labels=range(1,7),weights='quadratic'),f1_score(y,p,labels=range(1,7),average='macro',zero_division=0)])


def main():
    root=OUT/'statistics'
    a=pd.read_csv(STUDY/'runs/B_Matched_OFF_seed42/test_predictions.csv').sort_values('image_id')
    b=pd.read_csv(STUDY/'runs/B_Learnable_OFF_seed42/test_predictions.csv').sort_values('image_id')
    groups=sorted(a.group_id.unique());g=pd.Categorical(a.group_id,categories=groups).codes
    y=a.y_true.to_numpy();pa=a.y_pred.to_numpy();pb=b.y_pred.to_numpy();rng=np.random.default_rng(619118)
    for i in range(25):
        selection=rng.integers(0,len(groups),size=len(groups))
        idx=np.concatenate([np.flatnonzero(g==j) for j in selection])
        for p in [pa,pb]:
            cm=np.bincount((y[idx]-1)*6+p[idx]-1,minlength=36).reshape(6,6)
            assert np.allclose(cm_metrics(cm)[0],metric(y[idx],p[idx]),atol=1e-12,rtol=0)
        swap=rng.integers(0,2,len(groups)).astype(bool)[g]
        p1=np.where(swap,pb,pa);p2=np.where(swap,pa,pb)
        cm1=np.bincount((y-1)*6+p1-1,minlength=36).reshape(6,6)
        cm2=np.bincount((y-1)*6+p2-1,minlength=36).reshape(6,6)
        assert np.allclose(cm_metrics(cm1)[0]-cm_metrics(cm2)[0],metric(y,p1)-metric(y,p2),atol=1e-12,rtol=0)
    p=np.array([.01,.04,.03]);order=np.argsort(p);adjusted=np.empty(3)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(3,0,-1)))
    assert np.allclose(adjusted,[.03,.06,.06])
    query=pd.read_csv(root/'query_source_per_seed.csv');assert len(query)==39
    assert not query.duplicated(['stage','query','seed']).any()
    baseline=pd.read_csv(root/'final_baseline_comparison.csv');seeds=baseline[baseline.record_type=='seed']
    assert not seeds.duplicated(['model','seed']).any()
    assert len(seeds[seeds.model=='PGCA-0'])==3
    for seed in SEEDS:
        a=seeds[(seeds.model=='PGCA-0')&(seeds.seed==str(seed))].iloc[0]
        b=query[(query.stage=='A')&(query['query']=='Matched')&(query.seed==seed)].iloc[0]
        assert np.allclose(a[METRICS].to_numpy(float),b[METRICS].to_numpy(float),atol=1e-12)
    integrity=pd.read_csv(OUT/'audit/intervention_integrity.csv')
    assert len(integrity)==6 and (integrity.weight_sha256_before==integrity.weight_sha256_after).all()
    assert not integrity.gradient_enabled.any()
    maps=pd.read_csv(OUT/'evidence_archive/multi_donor_mappings.csv')
    assert len(maps)==3800
    for _,part in maps.groupby('donor_map'):
        assert part.recipient_group.nunique()==part.donor_group.nunique()==380
        assert (part.recipient_group!=part.donor_group).all()
    hashes=pd.read_csv(OUT/'evidence_archive/sha256_manifest.csv')
    for r in hashes.itertuples(): assert sha(r.path)==r.sha256,r.path
    lock={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write_json(OUT/'audit/analysis_source_lock.json',lock)
    report=dict(status='PASS',independent_bootstrap_metric_checks=50,independent_group_swap_checks=25,holm_known_case=True,
                unique_query_model_seed_rows=39,PGCA0_seed_rows=3,intervention_weights_unchanged=6,donor_mappings_verified=10,
                original_evidence_files_unchanged=len(hashes),manuscript_gate='await author claim confirmation')
    write_json(OUT/'audit/analysis_validation.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()

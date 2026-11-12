"""Frozen-output recomputation, cluster CIs and explicit permutation sensitivity."""
from common import *
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score, precision_recall_fscore_support
from statistics_common import cm_metrics, with_mean_sd, scalar_pair
from analyze_results import localization, LOC_METRICS, PAIR_LOC_METRICS

N = 10000


def read_prediction(path, reference):
    f=pd.read_csv(path).sort_values('image_id').reset_index(drop=True)
    r=reference.sort_values('image_id')
    assert len(f)==555 and f.image_id.nunique()==555 and f.group_id.nunique()==380
    for a,b in [('image_id','image_id'),('group_id','group_id'),('y_true','y')]:
        assert np.array_equal(f[a],r[b]),str(path)
    assert np.isin(f.y_pred,range(1,7)).all()
    return f


def metrics(f):
    y,p=f.y_true,f.y_pred
    vals=[accuracy_score(y,p),mean_absolute_error(y,p),cohen_kappa_score(y,p,labels=list(range(1,7)),weights='quadratic'),f1_score(y,p,labels=list(range(1,7)),average='macro',zero_division=0)]
    assert vals[1]>=1-vals[0]-1e-12
    return dict(zip(METRICS,vals))


def paired(left,right,name,permutation=False):
    reference=left[42].sort_values('image_id')
    groups=sorted(reference.group_id.unique())
    g=pd.Categorical(reference.group_id,categories=groups).codes
    w=np.random.default_rng(724299).multinomial(len(groups),np.ones(len(groups))/len(groups),size=N).astype(float)
    swaps=np.random.default_rng(20261103).integers(0,2,size=(N,len(groups))).astype(float) if permutation else None
    points,draws,nulls=[],[],[]
    for seed in SEEDS:
        cubes=[]
        for f in [left[seed],right[seed]]:
            f=f.sort_values('image_id')
            assert np.array_equal(f.image_id,reference.image_id) and np.array_equal(f.group_id,reference.group_id)
            assert np.array_equal(f.y_true,reference.y_true)
            bins=g*36+(f.y_true.to_numpy()-1)*6+f.y_pred.to_numpy()-1
            cubes.append(np.bincount(bins,minlength=len(groups)*36).reshape(len(groups),36).astype(float))
        a,b=cubes
        point=cm_metrics(a.sum(0).reshape(6,6))[0]-cm_metrics(b.sum(0).reshape(6,6))[0]
        draw=cm_metrics((w@a).reshape(N,6,6))[0]-cm_metrics((w@b).reshape(N,6,6))[0]
        points.append(point);draws.append(draw)
        if permutation:
            change=swaps@(b-a)
            null=cm_metrics((a.sum(0)+change).reshape(N,6,6))[0]-cm_metrics((b.sum(0)-change).reshape(N,6,6))[0]
            nulls.append(null)
    rows=[]
    for j,seed in enumerate(SEEDS+['mean_across_seeds']):
        point=points[j] if j<3 else np.mean(points,axis=0)
        draw=draws[j] if j<3 else np.mean(draws,axis=0)
        assert np.isfinite(draw).all()
        for k,m in enumerate(METRICS):
            low,high=np.quantile(draw[:,k],[.025,.975])
            row=dict(comparison=name,seed=seed,metric=m,difference=point[k],ci_low=low,ci_high=high,
                     difference_seed_SD=np.std(np.array(points)[:,k],ddof=1),
                     favorable_seeds=int(np.sum(np.array(points)[:,k]<0 if m=='MAE' else np.array(points)[:,k]>0)),
                     bootstrap_draws=N,groups=len(groups),images=len(reference),p_permutation=np.nan,p_Holm=np.nan)
            if permutation:
                null=nulls[j] if j<3 else np.mean(nulls,axis=0)
                row['p_permutation']=(1+np.sum(np.abs(null[:,k])>=abs(point[k])-1e-14))/(N+1)
            rows.append(row)
    return rows


def main():
    assert read_json(OUT/'audit/freeze_status.json')['status']=='PASS'
    torch.set_num_threads(4)
    manifest=pd.read_csv(BASE/'data/dataset_manifest.csv')
    test=manifest[manifest.split=='test']
    frozen=pd.read_csv(OUT/'evidence_archive/experiment_manifest.csv')
    pred,rows,classes={},[],[]
    for c in frozen.to_dict('records'):
        f=read_prediction(c['prediction'],test)
        key=(c['stage'],c['query'],c['seed']);pred[key]=f
        v=metrics(f)
        done=read_json(Path(c['prediction']).parent/'completed.json')
        assert all(abs(v[m]-done['test'][m])<1e-12 for m in METRICS)
        context={k:c[k] for k in ['stage','query','residual_on','lambda_ROI','seed']}
        prob=f[[f'p_ge_{k}' for k in range(2,7)]].to_numpy()
        assert np.array_equal(1+(prob>=.5).sum(1),f.y_pred)
        rows.append({**context,**v,**{k:c[k] for k in ['total','trainable','best_epoch','experiment_id']},
                     'nonmonotone_images':int((np.diff(prob,axis=1)>0).any(1).sum())})
        pr,re,fs,support=precision_recall_fscore_support(f.y_true,f.y_pred,labels=range(1,7),zero_division=0)
        classes.extend({**context,'BI_RADS':label,'Precision':p,'Recall':r,'F1':s,'Support':int(n)} for label,p,r,s,n in zip(['2','3','4A','4B','4C','5'],pr,re,fs,support))
    frame=pd.DataFrame(rows);classframe=pd.DataFrame(classes)
    frame.to_csv(OUT/'statistics/query_source_per_seed.csv',index=False)
    combined=with_mean_sd(frame,['stage','query','residual_on','lambda_ROI'],METRICS)
    combined[combined.record_type!='seed'].to_csv(OUT/'statistics/query_source_mean_std.csv',index=False)
    classframe.to_csv(OUT/'statistics/query_classwise_comparison.csv',index=False)
    with_mean_sd(classframe,['stage','query','BI_RADS'],['Precision','Recall','F1']).to_csv(OUT/'statistics/query_classwise_summary.csv',index=False)
    comparisons=[]
    for stage in ['A','B','C']:
        for control in ['Learnable','Shuffled','Sonographic','Constant']:
            if (stage,control,42) not in pred: continue
            comparisons.extend(paired({s:pred[(stage,'Matched',s)] for s in SEEDS},{s:pred[(stage,control,s)] for s in SEEDS},f'{stage}: Matched - {control}',stage=='B' and control!='Constant'))
            print('PAIRED',stage,control,flush=True)
    for query in ['Matched','Shuffled','Learnable','Sonographic','Constant']:
        comparisons.extend(paired({s:pred[('A',query,s)] for s in SEEDS},{s:pred[('B',query,s)] for s in SEEDS},f'{query}: Residual ON - OFF'))
    pairs=pd.DataFrame(comparisons)
    family=(pairs.seed=='mean_across_seeds')&pairs.comparison.isin([f'B: Matched - {q}' for q in ['Learnable','Shuffled','Sonographic']])
    indices=pairs[family].sort_values('p_permutation').index
    assert len(indices)==12
    adjusted=np.minimum(1,np.maximum.accumulate(pairs.loc[indices,'p_permutation'].to_numpy()*np.arange(12,0,-1)))
    pairs.loc[indices,'p_Holm']=adjusted
    pairs['holm_family']=family
    pairs.to_csv(OUT/'statistics/paired_query_comparisons.csv',index=False)

    # Recompute all new Stage C maps with the audited, unchanged spatial operator.
    locs=[];masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    for c in frozen.to_dict('records'):
        if c['stage']!='C': continue
        source=Path(c['prediction']).parent
        f=pd.read_csv(c['prediction'])
        loc=localization(c,f,np.load(c['attention']),manifest,masks)
        previous=pd.read_csv(source/'audited_localization_per_image.csv')
        assert np.allclose(loc[LOC_METRICS],previous[LOC_METRICS],atol=1e-12,rtol=0)
        locs.append(loc)
        print('ROI_RECOMPUTED',c['experiment_id'],flush=True)
    loc=pd.concat(locs,ignore_index=True)
    loc.to_csv(OUT/'statistics/localization_per_image.csv',index=False)
    locseed=loc.groupby(['stage','query','seed'])[LOC_METRICS].mean().reset_index()
    with_mean_sd(locseed,['stage','query'],LOC_METRICS).to_csv(OUT/'statistics/final_localization_comparison.csv',index=False)
    locpairs=[]
    for control in ['Learnable','Shuffled']:
        for metric in PAIR_LOC_METRICS:
            locpairs.extend(scalar_pair(loc[loc['query']=='Matched'],loc[loc['query']==control],metric,'C: Matched - '+control,replicates=N))
    for query in ['Matched','Shuffled','Learnable']:
        b=pd.concat([pd.read_csv(STUDY/'runs'/f'B_{query}_OFF_seed{s}'/'audited_localization_per_image.csv') for s in SEEDS])
        for metric in ['Pointing_Game','IoU_at_0.5','Top_10pct_IoU']:
            locpairs.extend(scalar_pair(loc[loc['query']==query],b,metric,query+': ROI-KL - no ROI (OFF)',replicates=N))
    pd.DataFrame(locpairs).to_csv(OUT/'statistics/localization_paired.csv',index=False)

    baselines={};base_rows=[]
    for path in sorted((BASE/'runs').glob('*/seed_*/test_predictions.csv')):
        config=read_json(path.parent/'config.json')
        name=config.get('model',config.get('model_name',path.parent.parent.name.replace('_','+')))
        scope='historical_standard'
        if config.get('permuted_training_pathology',False):
            name+='-train-only-permuted-pathology'
            scope='historical_one_seed_full_text_permutation_not_query_only'
        seed=int(config['seed'])
        f=read_prediction(path,test)
        assert seed not in baselines.get(name,{}),(name,seed)
        baselines.setdefault(name,{})[seed]=f
        base_rows.append(dict(model=name,seed=seed,**metrics(f),source=str(path),comparison_scope=scope,pretrained=False))
    strong=pd.read_csv(SUP/'strong_baselines.csv')
    for row in strong[strong.record_type=='seed'].to_dict('records'):
        path=Path(row['run_directory'])/'test_predictions.csv'
        f=read_prediction(path,test);seed=int(row['seed']);name=row['model']
        vals=metrics(f)
        assert all(abs(vals[m]-row[m])<1e-12 for m in METRICS)
        baselines.setdefault(name,{})[seed]=f
        base_rows.append(dict(model=name,seed=seed,**vals,source=str(path),comparison_scope='strong_baseline_validation_selected',pretrained=row['pretrained']))
    for path in sorted((BASE/'mechanism_validation_20261014/runs').glob('*/seed_*/test_predictions.csv')):
        name=path.parent.parent.name;seed=int(path.parent.name.split('_')[-1]);f=read_prediction(path,test)
        baselines.setdefault(name,{})[seed]=f
        base_rows.append(dict(model=name,seed=seed,**metrics(f),source=str(path),comparison_scope='mechanism_not_main_benchmark',pretrained=False))
    baseframe=pd.DataFrame(base_rows)
    assert not baseframe.duplicated(['model','seed']).any()
    assert len(baseframe[baseframe.model=='PGCA-0'])==3
    for seed in SEEDS:
        assert np.array_equal(baselines['PGCA-0'][seed].y_pred,pred[('A','Matched',seed)].y_pred)
    with_mean_sd(baseframe,['model','comparison_scope'],METRICS).to_csv(OUT/'statistics/final_baseline_comparison.csv',index=False)
    baseline_pairs=[]
    for left,right in [('PGCA-0','CatBoost-F'),('PGCA-full-SBpath','PGCA-0'),('PGCA-0','I+F+P-simple'),('PGCA-0','PGCA-0-uniform-QK')]:
        assert set(baselines[left])==set(baselines[right])==set(SEEDS),(left,right,list(baselines))
        baseline_pairs.extend(paired(baselines[left],baselines[right],left+' - '+right))
    pd.DataFrame(baseline_pairs).to_csv(OUT/'statistics/baseline_paired.csv',index=False)
    write_json(OUT/'audit/statistics_status.json',dict(status='PASS',recomputed_query_predictions=39,bootstrap_draws=N,permutation_draws=N,holm_family_tests=12,
               historical_prediction_files=len(base_rows),ROI_maps_recomputed=9,raw_metric_max_difference='less than 1e-12'))
    print('STATISTICS_PASS',flush=True)


if __name__=='__main__': main()

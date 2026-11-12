"""Evaluate the locked matrix, with paired group resampling and no model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'results_pgca36_pathology_20261013'
sys.path.insert(0,str(ROOT/'supplementary_validation_20261014'))
from statistics_common import METRICS, CLASSES, hard_metrics, classification_pair, scalar_pair, with_mean_sd

AREAS=[.01,.025,.05,.075,.1,.15,.2]
TOP_KEYS=[f'Top_{a*100:g}pct_IoU' for a in AREAS]
LOC_METRICS=['Pointing_Game','IoU_at_0.5','ROI_Coverage_at_0.5','Hot_Area_Fraction_at_0.5',*TOP_KEYS,
             *[f'Top_{a*100:g}pct_ROI_Coverage' for a in AREAS],*[f'Top_{a*100:g}pct_Hot_Area_Fraction' for a in AREAS]]
PAIR_LOC_METRICS=['Pointing_Game','IoU_at_0.5',*TOP_KEYS,'Top_10pct_ROI_Coverage','ROI_Coverage_at_0.5','Hot_Area_Fraction_at_0.5']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for b in iter(lambda:handle.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def table(frame,columns=None):
    frame=frame if columns is None else frame[columns]
    def text(v):
        return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(text(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def validate_metrics(frame):
    result=hard_metrics(frame)
    independently=[accuracy_score(frame.y_true,frame.y_pred),mean_absolute_error(frame.y_true,frame.y_pred),
                   cohen_kappa_score(frame.y_true,frame.y_pred,labels=list(range(1,7)),weights='quadratic'),
                   f1_score(frame.y_true,frame.y_pred,labels=list(range(1,7)),average='macro',zero_division=0)]
    assert np.allclose([result[m] for m in METRICS],independently,atol=1e-12,rtol=0)
    assert result['MAE']+1e-12>=1-result['Accuracy']
    assert np.isfinite(independently).all()
    return result


def overlap(selected,roi):
    intersection=np.logical_and(selected,roi).sum()
    union=np.logical_or(selected,roi).sum()
    return float(intersection/union),float(intersection/roi.sum()),float(selected.mean())


def localization(condition,predictions,maps,manifest,masks):
    test=manifest[manifest.split=='test']
    assert np.array_equal(test.image_id,predictions.image_id)
    assert maps.shape==(555,14,14) and np.isfinite(maps).all()
    assert np.allclose(maps.sum((1,2)),1,atol=1e-5)
    large=F.interpolate(torch.from_numpy(maps[:,None]),size=(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
    rows=[]
    for j,index in enumerate(test.index):
        roi=np.asarray(masks[index,0])>0
        if not roi.any(): continue
        heat=large[j]
        normalized=(heat-heat.min())/max(float(heat.max()-heat.min()),1e-12)
        peak=np.unravel_index(np.argmax(heat),heat.shape)
        iou,coverage,area=overlap(normalized>=.5,roi)
        row=dict(stage=condition['stage'],query=condition['query'],residual_on=condition['residual_on'],
                 seed=condition['seed'],lambda_ROI=condition['lambda_ROI'],image_id=predictions.image_id.iloc[j],
                 group_id=predictions.group_id.iloc[j],y_true=int(predictions.y_true.iloc[j]),y_pred=int(predictions.y_pred.iloc[j]),
                 Pointing_Game=int(roi[peak]),IoU_at_0_5=iou,ROI_area_fraction=float(roi.mean()),
                 ROI_Coverage_at_0_5=coverage,Hot_Area_Fraction_at_0_5=area,constant_map=bool(heat.max()==heat.min()))
        row['IoU_at_0.5']=row.pop('IoU_at_0_5')
        row['ROI_Coverage_at_0.5']=row.pop('ROI_Coverage_at_0_5')
        row['Hot_Area_Fraction_at_0.5']=row.pop('Hot_Area_Fraction_at_0_5')
        order=np.argsort(-heat.ravel(),kind='stable')
        for fraction in AREAS:
            k=int(np.ceil(fraction*heat.size))
            selected=np.zeros(heat.size,dtype=bool);selected[order[:k]]=True
            iou,coverage,area=overlap(selected.reshape(heat.shape),roi)
            prefix=f'Top_{fraction*100:g}pct'
            row[prefix+'_IoU']=iou
            row[prefix+'_ROI_Coverage']=coverage
            row[prefix+'_Hot_Area_Fraction']=area
            threshold=heat.ravel()[order[k-1]]
            row[prefix+'_boundary_tie_count']=int((heat==threshold).sum())
        rows.append(row)
    result=pd.DataFrame(rows)
    assert len(result)==383 and result.group_id.nunique()==290
    expected=pd.read_csv(BASE/'mechanism_validation_20261014/localization_roi_subset.csv')
    assert set(result.image_id)==set(expected.image_id)
    return result


def collect(conditions,manifest):
    rows,classrows,integrity,predictions,localizations=[],[],[],{},[]
    masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    (HERE/'analysis').mkdir(exist_ok=True)
    for c in conditions:
        run=HERE/'runs'/c['run_id']
        config=json.loads((run/'config.json').read_text())
        complete=json.loads((run/'completed.json').read_text())
        curve=json.loads((run/'training_curve.json').read_text())
        pred=pd.read_csv(run/'test_predictions.csv')
        assert len(pred)==555 and pred.image_id.nunique()==555 and pred.group_id.nunique()==380
        probability=pred[[f'p_ge_{k}' for k in range(2,7)]].to_numpy()
        assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        assert np.array_equal(1+(probability>=.5).sum(1),pred.y_pred.to_numpy())
        expected=manifest[manifest.split=='test']
        for actual,reference in [(pred.image_id,expected.image_id),(pred.y_true,expected.y),(pred.group_id,expected.group_id)]:
            assert np.array_equal(actual.to_numpy(),reference.to_numpy())
        assert config['seed']==c['seed'] and config['query']==c['query'] and config['residual_on']==c['residual_on']
        assert digest(run/'split_manifest.csv')==digest(BASE/'split_manifest.csv')
        assert digest(run/'best_checkpoint.pt')==complete['checkpoint_sha256']
        assert len(curve)==30
        assert [r['epoch'] for r in curve]==list(range(1,31))
        selected=max(curve,key=lambda r:(r['validation']['QWK'],-r['validation']['MAE'],-r['epoch']))['epoch']
        assert selected==complete['best_epoch']
        losskey='training_loss' if c['reuse_candidate'] else 'loss'
        assert all(np.isfinite(r[losskey]) and np.isfinite(r['validation']['QWK']) for r in curve)
        values=validate_metrics(pred)
        for metric in METRICS: assert abs(values[metric]-complete['test'][metric])<1e-12
        cm=confusion_matrix(pred.y_true,pred.y_pred,labels=range(1,7))
        np.savetxt(run/'audited_confusion_matrix.csv',cm,fmt='%d',delimiter=',')
        context=dict(stage=c['stage'],query=c['query'],residual_on=c['residual_on'],lambda_ROI=c['lambda_ROI'],seed=c['seed'])
        fitted_commit=json.loads((Path(complete['source'])/'config.json').read_text())['code_commit'] if c['reuse_candidate'] else config['training_code_commit']
        rows.append({**context,**{m:values[m] for m in METRICS},'status':complete['status'],
                     'run_id':c['run_id'],'checkpoint_sha256':complete['checkpoint_sha256'],
                     'controlled_wrapper_commit':config['training_code_commit'],'fitted_training_commit':fitted_commit,
                     'best_epoch':selected,'test_images':555,'test_groups':380,**config['parameter_counts']})
        rows[-1]['threshold_monotonicity_violation_fraction']=float(np.any(np.diff(probability,axis=1)>1e-7,axis=1).mean())
        precision,recall,f1,support=precision_recall_fscore_support(pred.y_true,pred.y_pred,labels=range(1,7),zero_division=0)
        classrows.extend({**context,'BI_RADS':label,'Precision':float(p),'Recall':float(r),'F1':float(f),'Support':int(n)}
                         for label,p,r,f,n in zip(CLASSES,precision,recall,f1,support))
        integrity.append(dict(run_id=c['run_id'],status=complete['status'],epochs=30,checkpoint_selection_verified=True,
                              metrics_recomputed=True,MAE_bound_pass=True,split_sha256=digest(run/'split_manifest.csv'),
                              checkpoint_sha256=complete['checkpoint_sha256'],best_epoch=selected))
        predictions[(c['stage'],c['query'],c['seed'])]=pred
        cache=run/'audited_localization_per_image.csv'
        cache_key=run/'audited_localization_cache.json'
        expected_key=dict(analysis_sha256=digest(Path(__file__)),maps_sha256=digest(run/'test_attention_maps.npy'))
        if cache.exists() and cache_key.exists() and json.loads(cache_key.read_text())==expected_key:
            loc=pd.read_csv(cache)
        else:
            loc=localization(c,pred,np.load(run/'test_attention_maps.npy'),manifest,masks)
            loc.to_csv(cache,index=False)
            cache_key.write_text(json.dumps(expected_key))
        localizations.append(loc)
        print('AUDITED',c['run_id'],flush=True)
    return pd.DataFrame(rows),pd.DataFrame(classrows),pd.DataFrame(integrity),predictions,pd.concat(localizations,ignore_index=True)


def contrasts(predictions,localization_rows):
    definitions=[]
    for stage in ['A','B','C']:
        for query in ['Shuffled','Learnable','Sonographic','Constant']:
            if (stage,query,42) in predictions:
                primary=stage=='B' and query in ['Shuffled','Learnable','Sonographic']
                definitions.append(((stage,'Matched'),(stage,query),f'{stage}: Matched - {query}',primary,stage=='C' and query=='Learnable'))
    for query in ['Matched','Shuffled','Learnable','Sonographic','Constant']:
        if ('A',query,42) in predictions and ('B',query,42) in predictions:
            definitions.append((('A',query),('B',query),f'{query}: Residual ON - OFF',False,False))
    for query in ['Matched','Shuffled','Learnable']:
        if ('C',query,42) in predictions and ('B',query,42) in predictions:
            definitions.append((('C',query),('B',query),f'{query}: ROI-KL OFF - no-ROI OFF',False,False))
    output=[]
    for a,b,name,class_primary,loc_primary in definitions:
        left={s:predictions[(*a,s)] for s in [42,3407,2026]}
        right={s:predictions[(*b,s)] for s in [42,3407,2026]}
        rows=classification_pair(left,right,name,replicates=2000)
        for r in rows:
            r.update(domain='classification',prespecified_primary=class_primary,positive_favors='right' if r['metric']=='MAE' else 'left')
        output.extend(rows)
        l=localization_rows[(localization_rows['stage']==a[0])&(localization_rows['query']==a[1])]
        r=localization_rows[(localization_rows['stage']==b[0])&(localization_rows['query']==b[1])]
        assert len(l)==len(r)==383*3,(name,len(l),len(r))
        assert l.groupby('seed').size().eq(383).all() and r.groupby('seed').size().eq(383).all()
        for metric in PAIR_LOC_METRICS:
            rows=scalar_pair(l,r,metric,name,replicates=2000)
            for v in rows:
                v.update(domain='localization',prespecified_primary=loc_primary and metric=='Top_10pct_IoU',
                         positive_favors='neither_area_is_descriptive' if 'Area_Fraction' in metric else 'left')
            output.extend(rows)
        print('BOOTSTRAP',name,flush=True)
    result=pd.DataFrame(output)
    seed_rows=result[result.seed.astype(str)!='mean_across_seeds'].copy()
    seed_rows['favorable'] = np.where(seed_rows.metric=='MAE',seed_rows.difference<0,seed_rows.difference>0)
    variation=seed_rows.groupby(['comparison','domain','metric']).agg(
        difference_sample_SD=('difference',lambda x:x.std(ddof=1)),
        minimum_seed_difference=('difference','min'),maximum_seed_difference=('difference','max'),
        favorable_seed_count=('favorable','sum'),training_seeds=('seed','size')).reset_index()
    result=result.merge(variation,on=['comparison','domain','metric'],validate='many_to_one')
    result.loc[result.positive_favors=='neither_area_is_descriptive','favorable_seed_count']=np.nan
    return result


def map_pair_rows(a,b,identity):
    assert a.shape==b.shape and a.shape[-2:]==(14,14)
    rows=[]
    for aa,bb,context in zip(a,b,identity):
        aa,bb=aa.ravel().astype(float),bb.ravel().astype(float)
        constant=np.ptp(aa)==0 or np.ptp(bb)==0
        rho=float('nan') if constant else float(spearmanr(aa,bb).statistic)
        an,bn=aa/aa.sum(),bb/bb.sum()
        rows.append({**context,'Spearman_14x14':rho,'normalized_L1':float(np.abs(an-bn).sum()),
                     'constant_map_pair':constant,'max_abs_difference':float(np.abs(aa-bb).max())})
    return rows


def main(partial_stage=None):
    torch.set_num_threads(4)
    protocol=json.loads((HERE/'protocol.json').read_text())
    conditions=protocol['conditions']
    if partial_stage: conditions=[c for c in conditions if c['stage']==partial_stage]
    missing=[c['run_id'] for c in conditions if not (HERE/'runs'/c['run_id']/'completed.json').exists()]
    assert not missing,missing
    manifest=pd.read_csv(BASE/'data/dataset_manifest.csv')
    metrics,classes,integrity,predictions,loc=collect(conditions,manifest)
    group=['stage','query','residual_on','lambda_ROI']
    summary=with_mean_sd(metrics,group,METRICS)
    class_summary=with_mean_sd(classes,group+['BI_RADS'],['Precision','Recall','F1'])
    loc_seeds=loc.groupby(group+['seed'],sort=False)[LOC_METRICS].mean().reset_index()
    loc_summary=with_mean_sd(loc_seeds,group,LOC_METRICS)
    loc_summary['ROI_images']=383;loc_summary['ROI_groups']=290
    pairs=contrasts(predictions,loc)
    destination=HERE if not partial_stage else HERE/'analysis'/f'stage_{partial_stage}'
    destination.mkdir(exist_ok=True,parents=True)
    summary.to_csv(destination/'query_mechanism_results.csv',index=False)
    class_summary.to_csv(destination/'query_mechanism_classwise.csv',index=False)
    loc_summary.to_csv(destination/'roi_query_localization.csv',index=False)
    loc.to_csv(destination/'localization_per_image.csv',index=False)
    pairs.to_csv(destination/'paired_query_comparisons.csv',index=False)
    integrity.to_csv(destination/'run_integrity.csv',index=False)
    for stage in sorted(metrics.stage.unique()):
        result=summary[(summary.stage==stage)&(summary.record_type=='mean')]
        p=pairs[(pairs.comparison.str.startswith(stage+':'))&(pairs.seed.astype(str)=='mean_across_seeds')&(pairs.domain=='classification')]
        body=f'# Stage {stage} Results\n\nAll planned conditions for this stage completed. Values below are three-seed means; the CSV includes each seed and sample SD. No condition was selected on test performance.\n\n'
        body+=table(result,['query',*METRICS])+'\n## Paired Conditional Intervals\n\n'
        body+=table(p,['comparison','metric','difference','ci_low','ci_high'])
        if stage=='C':
            body+='\n## Equal ROI-KL Supervision\n\nAll methods use residual OFF, frozen ROI-KL weight0.01 and the identical383-image/290-group subset.\n\n'
            body+=table(loc_summary[(loc_summary.stage=='C')&(loc_summary.record_type=='mean')],
                        ['query','Pointing_Game','IoU_at_0.5','Top_10pct_IoU','ROI_Coverage_at_0.5','Hot_Area_Fraction_at_0.5'])
            relevant=pairs[(pairs.comparison.str.startswith('C:'))&(pairs.seed.astype(str)=='mean_across_seeds')&
                           (pairs.domain=='localization')&(pairs.metric.isin(['Pointing_Game','IoU_at_0.5','Top_10pct_IoU']))]
            body+='\n'+table(relevant,['comparison','metric','difference','ci_low','ci_high'])
        body+='\nDifferences are left minus right; lower MAE is favorable. Paired 2,000-draw risk-group intervals are conditional on these trained seeds and a fixed donor map, not independent confirmation or a replacement for seed SD.\n'
        (HERE/f'stage_{stage}_report.md').write_text(body)
    status=dict(completed_conditions=len(conditions),new_fits=int((metrics.status=='SUCCESS_NEW_FIT').sum()),
                reused=int((metrics.status=='SUCCESS_REUSED').sum()),metric_checks='PASS',localization_images_per_condition=383,
                constant_map_count=int(loc.constant_map.sum()),max_top10_boundary_ties=int(loc.Top_10pct_boundary_tie_count.max()),
                bootstrap_replicates=2000,analysis_complete=not bool(partial_stage))
    (destination/'analysis_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial-stage',choices=['A','B','C']);args=p.parse_args()
    main(args.partial_stage)

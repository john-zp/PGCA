"""Cluster uncertainty preserves donor and training-seed repeated measures."""
from common import *
import numpy as np
import pandas as pd

VALUES=['attention_TV','Spearman','top10_native_Jaccard','max_attention_shift',
        'mean_abs_threshold_shift','max_abs_threshold_shift','soft_ordinal_index_shift',
        'abs_soft_ordinal_index_shift','hard_changed','max_threshold_probability_shift',
        'nonmonotone_before','nonmonotone_after','identical_donor_embedding']


def summarize(part,context):
    # Collapse repeated donors/seeds within image before resampling patient-risk groups.
    images=part.groupby(['image_id','group_id'],sort=True)[VALUES].mean().reset_index()
    groups=sorted(images.group_id.unique())
    sums=images.groupby('group_id')[VALUES].sum().reindex(groups)
    counts=images.groupby('group_id')[VALUES].count().reindex(groups)
    w=np.random.default_rng(724299).multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000).astype(float)
    den=w@counts.to_numpy();num=w@sums.to_numpy()
    draws=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    rows=[]
    for j,v in enumerate(VALUES):
        low,high=np.nanquantile(draws[:,j],[.025,.975])
        group_mean=images.groupby('group_id')[v].mean()
        rows.append({**context,'metric':v,'image_weighted_mean':images[v].mean(),
                     'risk_group_equal_mean':group_mean.mean(),'ci_low':low,'ci_high':high,
                     'images':len(images),'groups':len(groups),'missing_images':int(images[v].isna().sum()),
                     'bootstrap_draws':10000,'global_max':images[v].max()})
    return rows


def main():
    single=pd.read_csv(OUT/'statistics/query_only_intervention_predictions.csv')
    multi=pd.read_csv(OUT/'statistics/multi_donor_per_image.csv')
    assert len(single)==3330 and len(multi)==33300
    for name,data in [('query_probability_intervention',single),('multi_donor_intervention',multi)]:
        rows=[]
        for (stage,seed,donor),part in data.groupby(['stage','seed','donor_map']):
            assert len(part)==555
            rows.extend(summarize(part,dict(stage=stage,seed=str(seed),donor_map=donor)))
        if name=='multi_donor_intervention':
            for (stage,seed),part in data.groupby(['stage','seed']):
                rows.extend(summarize(part,dict(stage=stage,seed=str(seed),donor_map='mean_10_donors')))
        for stage,part in data.groupby('stage'):
            rows.extend(summarize(part,dict(stage=stage,seed='mean_3_seeds',donor_map='mean_10_donors' if name.startswith('multi') else 'historical_single')))
        summary=pd.DataFrame(rows)
        summary.to_csv(OUT/f'statistics/{name}.csv',index=False)
        data.groupby(['stage','seed','donor_map','group_id'])[VALUES].mean().reset_index().to_csv(OUT/f'statistics/{name}_risk_groups.csv',index=False)
    variation=multi.groupby(['stage','seed','donor_map'])[VALUES].mean().reset_index()
    variation.to_csv(OUT/'statistics/donor_means_per_seed.csv',index=False)
    variation.groupby(['stage','seed'])[VALUES].agg(['mean','std']).to_csv(OUT/'statistics/donor_variability.csv')
    mono=pd.concat([single.assign(scope='historical_single'),multi.assign(scope='ten_new_donors')])
    mono.groupby(['stage','seed','scope']).agg(before_nonmonotone=('nonmonotone_before','mean'),after_nonmonotone=('nonmonotone_after','mean'),
                hard_change_rate=('hard_changed','mean'),mean_abs_threshold_shift=('mean_abs_threshold_shift','mean'),
                maximum_any_threshold_shift=('max_abs_threshold_shift','max')).reset_index().to_csv(OUT/'statistics/ordinal_probability_audit.csv',index=False)
    threshold=[]
    for scope,data in [('historical_single',single),('ten_new_donors',multi)]:
        for (stage,seed),part in data.groupby(['stage','seed']):
            for k in range(2,7):
                threshold.append(dict(scope=scope,stage=stage,seed=seed,threshold=k,mean_before=part[f'p_before_ge_{k}'].mean(),
                                      mean_after=part[f'p_after_ge_{k}'].mean(),mean_signed_shift=part[f'p_shift_ge_{k}'].mean(),
                                      mean_absolute_shift=part[f'p_shift_ge_{k}'].abs().mean()))
    pd.DataFrame(threshold).to_csv(OUT/'statistics/ordinal_threshold_summary.csv',index=False)
    report='# Ordinal Probability Interpretation\n\n'
    report+='The fitted head trains five independent binary logits against 1[Y>=k], k=2,...,6 for unit-spaced codes1,...,6. Inference is 1+count(sigmoid(logit)>=0.5). There is no monotone projection, chain rule, softmax or categorical calibration. Six-class probabilities are not a model output.\n\n'
    report+='The audits count any strict increase between adjacent cumulative threshold probabilities. Nonmonotone outputs prevent a globally valid six-class cumulative distribution. We do not repair or normalize them. Categorical maximum confidence, Jensen-Shannon divergence and a validated expected category are therefore NA, not zero.\n\n'
    report+='The computable soft ordinal index s=1+sum(p_k) is retained for every image. For a coherent cumulative distribution this identity equals its expected encoded grade; here it is explicitly an uncalibrated sum score, not evidence of a calibrated clinical BI-RADS probability distribution. The named binary-threshold maximum and certainty fields are not six-class confidence.\n\n'
    report+='For each intervention, all five original/altered logits and sigmoid values are saved. Hard changes are reported alongside continuous shifts. Total variation is half the historical sum-L1, and the two quantities are not mixed.\n\n'
    report+='Ten donor outcomes are averaged within each image before pooled group intervals; the same risk-group sample applies to all seeds/maps. Donor SD and training-seed variation are separate. Group-equal descriptive means are also shown; primary CIs target the image-weighted mean. Undefined Spearman values remain missing and their counts are recorded.\n'
    (OUT/'audit/02_ordinal_intervention_report.md').write_text(report)
    write_json(OUT/'audit/intervention_summary_status.json',dict(status='PASS',single_rows=len(single),multi_rows=len(multi),categorical_JS='not defined by fitted model',soft_index='1+sum of independent threshold probabilities'))


if __name__=='__main__': main()

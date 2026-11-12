"""Post hoc numerical QA of tie ambiguity; primary ranking rule stays unchanged."""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from analyze_results import HERE, BASE, AREAS, scalar_pair, table


def bounds(sorted_negative,ordered_roi,k):
    threshold=sorted_negative[k-1]
    left=int(np.searchsorted(sorted_negative,threshold,side='left'))
    right=int(np.searchsorted(sorted_negative,threshold,side='right'))
    prefix=np.concatenate([[0],np.cumsum(ordered_roi)])
    strict=int(prefix[left]);tied_roi=int(prefix[right]-prefix[left]);ties=right-left;remaining=k-left
    low=strict+max(0,remaining-(ties-tied_roi));high=strict+min(remaining,tied_roi)
    area=int(prefix[-1])
    return low/(k+area-low),high/(k+area-high),ties,remaining


def main():
    torch.set_num_threads(2)
    # Four tied cells, two ROI cells, choose two: IoU can range from zero to one.
    assert bounds(np.zeros(4),np.array([1,1,0,0]),2)==(0.,1.,4,2)
    protocol=json.loads((HERE/'protocol.json').read_text())
    manifest=pd.read_csv(BASE/'data/dataset_manifest.csv')
    test=manifest[manifest.split=='test']
    masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    records=[]
    for c in protocol['conditions']:
        run=HERE/'runs'/c['run_id']
        stable=pd.read_csv(run/'audited_localization_per_image.csv').set_index('image_id')
        maps=np.load(run/'test_attention_maps.npy')
        large=F.interpolate(torch.from_numpy(maps[:,None]),(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
        for j,(index,source) in enumerate(test.iterrows()):
            roi=np.asarray(masks[index,0])>0
            if not roi.any(): continue
            heat=large[j].ravel();mask=roi.ravel()
            order=np.argsort(-heat,kind='stable');negative=-heat[order];ordered_roi=mask[order]
            peak_ties=mask[heat==heat.max()]
            for fraction in AREAS:
                k=int(np.ceil(fraction*heat.size))
                lo,hi,ties,remaining=bounds(negative,ordered_roi,k)
                actual=float(stable.loc[source.image_id,f'Top_{fraction*100:g}pct_IoU'])
                assert lo-1e-10<=actual<=hi+1e-10
                records.append(dict(stage=c['stage'],query=c['query'],seed=c['seed'],image_id=source.image_id,group_id=source.group_id,
                                    fraction=fraction,stable_IoU=actual,tie_lower_IoU=lo,tie_upper_IoU=hi,
                                    boundary_tie_count=ties,selected_tied_pixels=remaining,
                                    Pointing_lower=float(peak_ties.all()),Pointing_upper=float(peak_ties.any())))
        print('TIE_QA',c['run_id'],flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/'localization_tie_per_image.csv',index=False)
    summary=frame.groupby(['stage','query','seed','fraction']).agg(stable_IoU=('stable_IoU','mean'),
        tie_lower_IoU=('tie_lower_IoU','mean'),tie_upper_IoU=('tie_upper_IoU','mean'),
        Pointing_lower=('Pointing_lower','mean'),Pointing_upper=('Pointing_upper','mean'),
        maximum_boundary_ties=('boundary_tie_count','max')).reset_index()
    summary['tie_ambiguity_span']=summary.tie_upper_IoU-summary.tie_lower_IoU
    summary.to_csv(HERE/'localization_tie_sensitivity.csv',index=False)
    comparisons=[]
    for query in ['Learnable','Shuffled']:
        left=frame[(frame['stage']=='C')&(frame['query']=='Matched')&(frame.fraction==.1)]
        right=frame[(frame['stage']=='C')&(frame['query']==query)&(frame.fraction==.1)]
        for name,lkey,rkey in [('lower_bound','tie_lower_IoU','tie_upper_IoU'),('upper_bound','tie_upper_IoU','tie_lower_IoU')]:
            a=left[['image_id','group_id','seed',lkey]].rename(columns={lkey:'bound_value'})
            b=right[['image_id','group_id','seed',rkey]].rename(columns={rkey:'bound_value'})
            rows=scalar_pair(a,b,'bound_value',f'C Matched-{query}: {name}',replicates=2000)
            for r in rows:r['bound_type']=name
            comparisons.extend(rows)
    comp=pd.DataFrame(comparisons);comp.to_csv(HERE/'localization_tie_bound_comparisons.csv',index=False)
    main_rows=comp[comp.seed.astype(str)=='mean_across_seeds']
    report='# Localization Tie Sensitivity\n\n'
    report+='This post hoc numerical QA was added after StageA revealed large equal-valued backgrounds in some maps. It does not change any primary Top-k mask, threshold, model, test subset or training choice. The primary result always uses the frozen stable row-major tie rule.\n\n'
    report+='At each of all seven fixed areas, strict-above-boundary pixels are fixed and the remaining tied pixels are bounded by choosing the fewest/most ROI pixels possible. These are deterministic ambiguity bounds, not optimized reported masks or confidence intervals. The full per-image and per-seed tables retain both bounds. Pointing-game ties are likewise bounded by whether none/all/some maximal pixels lie in the rectangle.\n\n'
    report+='For the main equal-ROI-KL Top-10% comparisons, the following bootstrap intervals concern the lower-bound or upper-bound **difference**, not a replacement estimate of the primary effect. This distinguishes numerical tie sensitivity from sampling uncertainty.\n\n'
    report+=table(main_rows,['comparison','difference','ci_low','ci_high'])
    report+='\nAny favorable comparison that changes direction within these bounds should not be described as tie-robust. No assumption that Matched is superior is used in deriving either bound.\n'
    (HERE/'localization_tie_sensitivity.md').write_text(report)
    (HERE/'analysis/tie_sensitivity_status.json').write_text(json.dumps(dict(status='PASS',conditions=39,areas=7,per_image_rows=len(frame))))


if __name__=='__main__': main()

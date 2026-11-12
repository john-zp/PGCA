"""Map sensitivity: trained controls and a separate Query-only intervention."""
import json
import pandas as pd
from torch.utils.data import DataLoader
from query_model import *
from analyze_results import map_pair_rows, table


def main():
    torch.set_num_threads(8)
    protocol=json.loads((HERE/'protocol.json').read_text())
    rows=[]
    for stage in ['A','B','C']:
        for seed in [42,3407,2026]:
            c=next(c for c in protocol['conditions'] if c['stage']==stage and c['query']=='Matched' and c['seed']==seed)
            run=HERE/'runs'/c['run_id']
            own=np.load(run/'test_attention_maps.npy')
            pred=pd.read_csv(run/'test_predictions.csv')
            masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
            test=QueryCohort('test',seed,'Matched')
            identities=[dict(stage=stage,seed=seed,image_id=r.image_id,group_id=r.group_id,valid_ROI=bool(np.any(masks[index])))
                        for r,index in zip(pred.itertuples(),test.indices)]
            for query in ['Shuffled','Learnable']:
                other=next(k for k in protocol['conditions'] if k['stage']==stage and k['query']==query and k['seed']==seed)
                maps=np.load(HERE/'runs'/other['run_id']/'test_attention_maps.npy')
                identity=[{**i,'comparison':f'trained Matched vs trained {query}','kind':'separate_trained_models'} for i in identities]
                rows.extend(map_pair_rows(own,maps,identity))
            set_seed(seed)
            model=make_query_model('Matched',c['residual_on'],seed).cuda().eval()
            model.load_state_dict(torch.load(run/'best_checkpoint.pt',map_location='cuda',weights_only=False)['model'])
            donor_test=QueryCohort('test',seed,'Shuffled')
            changed_maps=[]
            changed_predictions=[]
            with torch.no_grad():
                for batch_number,data in enumerate(DataLoader(donor_test,batch_size=16,shuffle=False)):
                    if batch_number==0:
                        model.inspect_paths=True
                        matched=model(**inputs(data,with_label=False))
                    model.query_source='Shuffled'
                    changed=model(**inputs(data,with_label=False))
                    if batch_number==0:
                        for field in ['residual','image_global','structured']:
                            assert torch.equal(matched['pathways'][field],changed['pathways'][field])
                        assert not torch.equal(matched['pathways']['query'],changed['pathways']['query'])
                        model.inspect_paths=False
                    changed_maps.append(changed['attn_map'].reshape(-1,14,14).cpu().numpy())
                    changed_predictions.extend(changed['pred'].cpu().tolist())
            changed_maps=np.concatenate(changed_maps)
            identity=[{**i,'comparison':'same checkpoint: own Query vs donor Query','kind':'query_only_inference_intervention',
                       'prediction_changed':int(p!=q)} for i,p,q in zip(identities,pred.y_pred,changed_predictions)]
            rows.extend(map_pair_rows(own,changed_maps,identity))
            np.save(run/'query_only_donor_intervention_maps.npy',changed_maps)
            print('SENSITIVITY',stage,seed,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'attention_map_sensitivity_per_image.csv',index=False)
    summary=frame.groupby(['stage','seed','kind','comparison'],dropna=False).agg(
        mean_Spearman=('Spearman_14x14','mean'),median_Spearman=('Spearman_14x14','median'),
        mean_normalized_L1=('normalized_L1','mean'),median_normalized_L1=('normalized_L1','median'),
        mean_max_abs_difference=('max_abs_difference','mean'),constant_map_pairs=('constant_map_pair','sum'),
        prediction_change_rate=('prediction_changed','mean'),images=('image_id','size')).reset_index()
    summary.to_csv(HERE/'attention_map_sensitivity_summary.csv',index=False)
    report='# Attention Map Sensitivity\n\n'
    report+='Native 14x14 maps are compared on all 555 identical test images. Spearman uses average ranks; undefined correlations for constant maps remain missing, never zero. Normalized L1 compares unit-mass maps.\n\n'
    report+='Separately trained controls differ in fitted parameters and therefore do not isolate an instantaneous Query effect. The within-checkpoint intervention changes only the Query donor, verifies the matched residual/global-image/structured representations remain identical, and retains the same weights. It is an inference sensitivity diagnostic, not a substitute for the full Shuffled training condition and not a causal explanation-fidelity test.\n\n'
    report+=table(summary)
    (HERE/'attention_map_sensitivity.md').write_text(report)


if __name__=='__main__': main()

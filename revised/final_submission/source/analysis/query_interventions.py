"""Frozen-weight, Query-only interventions. No optimizer or training API."""
from common import *
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from query_model import QueryCohort, make_query_model, set_seed, inputs


def model_digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_maps(test):
    groups=sorted(test.group_id.unique())
    representatives={g:min(f.index,key=lambda i:hashlib.sha256(('final_donor_rep_v1:'+str(f.loc[i,'image_id'])).encode()).hexdigest()) for g,f in test.groupby('group_id')}
    arrays=[];rows=[]
    for draw in range(10):
        rng=np.random.default_rng(202611030+draw)
        for attempt in range(10000):
            permutation=rng.permutation(len(groups))
            if np.all(permutation!=np.arange(len(groups))): break
        else: raise AssertionError('No derangement')
        mapping={g:groups[permutation[i]] for i,g in enumerate(groups)}
        assert len(set(mapping.values()))==len(groups)
        indices=np.array([representatives[mapping[g]] for g in test.group_id])
        arrays.append(indices)
        for recipient,donor in mapping.items():
            rows.append(dict(donor_map=draw,map_seed=202611030+draw,recipient_group=recipient,donor_group=donor,
                             donor_index=representatives[donor],donor_image_id=test.loc[representatives[donor],'image_id']))
    pd.DataFrame(rows).to_csv(OUT/'evidence_archive/multi_donor_mappings.csv',index=False)
    np.save(OUT/'evidence_archive/multi_donor_indices.npy',np.stack(arrays))
    write_json(OUT/'evidence_archive/multi_donor_design.json',dict(map_seeds=list(range(202611030,202611040)),
               map_count=10,unit='test risk group',no_replacement=True,no_self_match=True,label_blind=True,
               identical_embedding_donors='retained; no semantic filtering',maps_sha256=sha(OUT/'evidence_archive/multi_donor_indices.npy')))
    return arrays


def map_scores(a,b):
    a=a.reshape(len(a),-1).astype(float);b=b.reshape(len(b),-1).astype(float)
    a=a/a.sum(1,keepdims=True);b=b/b.sum(1,keepdims=True)
    ra=rankdata(a,axis=1);rb=rankdata(b,axis=1)
    ra-=ra.mean(1,keepdims=True);rb-=rb.mean(1,keepdims=True)
    den=np.sqrt((ra**2).sum(1)*(rb**2).sum(1))
    rho=np.divide((ra*rb).sum(1),den,out=np.full(len(a),np.nan),where=den>0)
    sa=np.argsort(-a,axis=1,kind='stable')[:,:20];sb=np.argsort(-b,axis=1,kind='stable')[:,:20]
    overlap=np.array([len(set(x)&set(y)) for x,y in zip(sa,sb)])
    return dict(attention_TV=.5*np.abs(a-b).sum(1),Spearman=rho,top10_native_Jaccard=overlap/(40-overlap),
                max_attention_shift=np.abs(a-b).max(1),constant_map_pair=den==0)


def metrics_rows(context,identities,p0,p1,l0,l1,a0,a1,donors,own_embedding,donor_embedding):
    delta=p1-p0;old=1+(p0>=.5).sum(1);new=1+(p1>=.5).sum(1)
    values=dict(**map_scores(a0,a1),mean_abs_threshold_shift=np.abs(delta).mean(1),max_abs_threshold_shift=np.abs(delta).max(1),
                soft_ordinal_index_before=1+p0.sum(1),soft_ordinal_index_after=1+p1.sum(1),
                soft_ordinal_index_shift=delta.sum(1),abs_soft_ordinal_index_shift=np.abs(delta.sum(1)),
                hard_before=old,hard_after=new,hard_changed=(old!=new).astype(int),
                max_threshold_probability_before=p0.max(1),max_threshold_probability_after=p1.max(1),
                max_threshold_probability_shift=p1.max(1)-p0.max(1),
                binary_certainty_before=np.maximum(p0,1-p0).max(1),binary_certainty_after=np.maximum(p1,1-p1).max(1),
                nonmonotone_before=(np.diff(p0,axis=1)>0).any(1),nonmonotone_after=(np.diff(p1,axis=1)>0).any(1),
                identical_donor_embedding=(own_embedding==donor_embedding).all(1),donor_index=donors)
    frame=pd.DataFrame({**identities,**{k:np.repeat(v,len(p0)) for k,v in context.items()},**values})
    for k in range(5):
        for prefix,arr in [('p_before',p0),('p_after',p1),('logit_before',l0),('logit_after',l1),('p_shift',delta)]:
            frame[f'{prefix}_ge_{k+2}']=arr[:,k]
    frame['categorical_max_confidence_before']=np.nan
    frame['categorical_max_confidence_after']=np.nan
    frame['categorical_JS']=np.nan
    frame['expected_grade_before']=np.nan
    frame['expected_grade_after']=np.nan
    frame['probability_limitation']='No categorical distribution in fitted independent-threshold implementation; soft ordinal index is separate.'
    return frame


def main():
    assert read_json(OUT/'audit/freeze_status.json')['status']=='PASS'
    assert torch.cuda.is_available(), 'Inference GPU unavailable; no training fallback'
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    manifest=pd.read_csv(BASE/'data/dataset_manifest.csv');test=manifest[manifest.split=='test']
    new_maps=make_maps(test)
    old=np.load(STUDY/'design/query_donors.npy')[test.index]
    donors=[old,*new_maps];donor_ids=['historical_single',*[f'map_{i:02}' for i in range(10)]]
    embeddings=np.load(BASE/'data/pathology_embeddings.npy',mmap_mode='r')
    masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    ownemb=np.asarray(embeddings[test.index]);allrows=[];checks=[]
    cases=read_json(STUDY/'design/visualization_cases.json')['cases']
    casepositions=[list(test.index).index(c['index']) for c in cases]
    for stage in ['A','B']:
        for seed in SEEDS:
            run=STUDY/'runs'/f'{stage}_Matched_{"ON" if stage=="A" else "OFF"}_seed{seed}'
            set_seed(seed);model=make_query_model('Matched',stage=='A',seed).cuda().eval()
            checkpoint=torch.load(run/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model'],strict=True)
            model.inspect_paths=True
            before_digest=model_digest(model)
            dataset=QueryCohort('test',seed,'Matched')
            ps=[[] for _ in range(12)];ls=[[] for _ in range(12)];maps=[[] for _ in range(12)];images=[]
            cursor=0
            for data in DataLoader(dataset,batch_size=16,shuffle=False):
                n=len(data['image']);model.query_source='Matched'
                args=inputs(data,with_label=False)
                original=model(**args)
                images.append(data['image'].numpy())
                outputs=[original]
                for donor in donors:
                    model.query_source='Shuffled'
                    donor_emb=torch.from_numpy(np.array(embeddings[donor[cursor:cursor+n]],copy=True)).cuda()
                    changed=model(**args,query_emb=donor_emb)
                    for key in ['residual','image_global','structured']:
                        assert torch.equal(original['pathways'][key],changed['pathways'][key]),key
                    if stage=='B': assert not torch.count_nonzero(changed['pathways']['residual'])
                    outputs.append(changed)
                for i,out in enumerate(outputs):
                    assert torch.isfinite(out['p_ge']).all() and torch.isfinite(out['attn_map']).all()
                    ps[i].append(out['p_ge'].cpu().numpy());ls[i].append(out['logits'].cpu().numpy())
                    maps[i].append(out['attn_map'].reshape(-1,14,14).cpu().numpy())
                cursor+=n
            p=np.stack([np.concatenate(v) for v in ps]);logits=np.stack([np.concatenate(v) for v in ls]);a=np.stack([np.concatenate(v) for v in maps])
            original_prediction=pd.read_csv(run/'test_predictions.csv')
            assert np.allclose(p[0],original_prediction[[f'p_ge_{k}' for k in range(2,7)]],atol=1e-7,rtol=0)
            assert np.array_equal(a[0],np.load(run/'test_attention_maps.npy'))
            assert np.array_equal(a[1],np.load(run/'query_only_donor_intervention_maps.npy'))
            assert model_digest(model)==before_digest
            destination=OUT/'statistics/intervention_arrays';destination.mkdir(exist_ok=True)
            np.savez_compressed(destination/f'{stage}_seed{seed}.npz',probabilities=p,logits=logits,attention=a,
                                test_indices=test.index.to_numpy(),donor_ids=np.array(donor_ids))
            identities=dict(image_id=test.image_id.to_numpy(),group_id=test.group_id.to_numpy(),y_true=test.y.to_numpy())
            for j,name in enumerate(donor_ids):
                allrows.append(metrics_rows(dict(stage=stage,residual_on=stage=='A',seed=seed,donor_map=name),identities,p[0],p[j+1],logits[0],logits[j+1],a[0],a[j+1],donors[j],ownemb,np.asarray(embeddings[donors[j]])))
            if seed==42:
                np.savez_compressed(destination/f'{stage}_visual_cases.npz',images=np.concatenate(images)[casepositions],roi=np.asarray(masks[test.index[casepositions]]),
                                    attention=a[:,:,][...,],case_positions=np.array(casepositions),y_true=test.y.iloc[casepositions].to_numpy(),probabilities=p)
            checks.append(dict(stage=stage,seed=seed,weight_sha256_before=before_digest,weight_sha256_after=model_digest(model),checkpoint_sha256=sha(run/'best_checkpoint.pt'),
                               original_map_replay_exact=True,prior_donor_map_replay_exact=True,all_batch_other_paths_exact=True,gradient_enabled=torch.is_grad_enabled()))
            print('INTERVENTION',stage,seed,flush=True)
    result=pd.concat(allrows,ignore_index=True)
    result[result.donor_map=='historical_single'].to_csv(OUT/'statistics/query_only_intervention_predictions.csv',index=False)
    result[result.donor_map!='historical_single'].to_csv(OUT/'statistics/multi_donor_per_image.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'audit/intervention_integrity.csv',index=False)
    write_json(OUT/'audit/intervention_status.json',dict(status='PASS',checkpoints=6,new_donor_maps=10,images_per_run=555,groups=380,trained=False,updated_weights=False))
    print('INTERVENTIONS_PASS',flush=True)


if __name__=='__main__': main()

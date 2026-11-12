"""Pathway interventions and numerical checks before any formal fit."""
import copy
import io
import json
import time
import traceback

from query_model import *
from torch.utils.data import DataLoader


def norm(values):
    return sum(float(v.detach().double().square().sum()) for v in values if v is not None)**.5


def main():
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    protocol=json.loads((HERE/'protocol.json').read_text())
    results=[]
    reference_paths={}
    for condition in protocol['conditions']:
        seed,query,on=condition['seed'],condition['query'],condition['residual_on']
        dataset=QueryCohort('train',seed,query)
        batch=next(iter(DataLoader(dataset,batch_size=4,shuffle=False)))
        set_seed(seed)
        model=make_query_model(query,on,seed).eval()
        counts=parameter_counts(model)
        model.inspect_paths=True
        out=model(**inputs(batch,'cpu'))
        assert out['logits'].shape==(4,5) and out['fusion_out'].shape==(4,256)
        assert out['attn_map'].shape==(4,1,196) and out['grid_hw']==(14,14)
        assert torch.isfinite(out['logits']).all() and torch.isfinite(out['attn_map']).all()
        assert torch.allclose(out['attn_map'].sum(-1),torch.ones(4,1),atol=1e-6,rtol=0)
        path={k:v.detach().clone() for k,v in out['pathways'].items()}
        if query=='Matched':
            reference_paths[seed,on]=path
        for key in ['image_global','structured','residual']:
            assert torch.equal(path[key],reference_paths[seed,on][key]), (condition,key)
        if not on:
            assert torch.count_nonzero(path['residual'])==0
        if query in ['Constant','Learnable']:
            assert torch.equal(path['query'],path['query'][0:1].expand_as(path['query']))
        if query=='Shuffled':
            assert not torch.equal(path['query'],reference_paths[seed,on]['query'])
        if query=='Sonographic':
            altered=inputs(batch,'cpu')
            altered['features']=1-altered['features']
            assert not torch.equal(model(**altered)['pathways']['query'],path['query'])
        if query=='Matched' and on:
            set_seed(seed)
            original=make_model('PGCA-0').eval()
            assert all(torch.equal(v,original.state_dict()[k]) for k,v in model.state_dict().items())
            baseline=original(**inputs(batch,'cpu'))
            assert torch.equal(out['logits'],baseline['logits'])
            assert torch.equal(out['attn_map'],baseline['attn_map'])
        changed=inputs(batch,'cpu')
        changed['phrase_emb']=changed['phrase_emb'].flip(0)+.125
        if not on and query in ['Learnable','Sonographic','Constant','Shuffled']:
            changed_out=model(**changed)
            assert torch.equal(out['logits'],changed_out['logits'])
            assert torch.equal(out['attn_map'],changed_out['attn_map'])
        # Blocking Q's gradient must remove every remaining text-to-prediction path in OFF.
        b=inputs(batch,'cpu')
        b['phrase_emb']=b['phrase_emb'].clone().requires_grad_(True)
        if 'query_emb' in b: b['query_emb']=b['query_emb'].clone().requires_grad_(True)
        handle=model.cross_attn.mha.register_forward_pre_hook(lambda module,args:(args[0].detach(),)+args[1:])
        detached=model(**b)
        text_tensors=[b['phrase_emb']]+([b['query_emb']] if 'query_emb' in b else [])
        direct=norm(torch.autograd.grad(detached['loss_ord'],text_tensors,allow_unused=True))
        handle.remove()
        assert (direct>0) if on else (direct==0), (condition,direct)
        if not on and query in ['Matched','Shuffled']:
            b=inputs(batch,'cpu')
            key='query_emb' if query=='Shuffled' else 'phrase_emb'
            b[key]=b[key].clone().requires_grad_(True)
            connected=model(**b)
            query_gradient=norm(torch.autograd.grad(connected['loss_ord'],[b[key]],allow_unused=True))
            assert query_gradient>0
        else: query_gradient=None
        model=model.cuda().train()
        model.inspect_paths=False
        fixed=model.frozen_query.detach().clone() if query=='Constant' else None
        learned=model.learnable_query.detach().clone() if query=='Learnable' else None
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001)
        grads=[]
        for i,data in enumerate(DataLoader(dataset,batch_size=4,shuffle=False)):
            opt.zero_grad(set_to_none=True)
            out=model(**inputs(data))
            loss,kl,details=objective(out,data['roi'].cuda(),condition['lambda_ROI'])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            attention_norm=norm([p.grad for n,p in model.named_parameters() if n.startswith('cross_attn.')])
            assert attention_norm>0
            if condition['lambda_ROI']:
                assert details['alpha_mass_error']<1e-6 and details['mu_mass_error']<1e-6
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            grads.append({'loss':float(loss.detach()),'attention_gradient_norm':attention_norm,'roi_kl':float(kl.detach()),**details})
            if i==1: break
        if fixed is not None: assert torch.equal(fixed,model.frozen_query)
        if learned is not None: assert not torch.equal(learned,model.learnable_query)
        model.eval()
        expected=model(**inputs(batch))['logits'].detach()
        buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True),strict=True)
        assert torch.equal(expected,model(**inputs(batch))['logits'])
        results.append({**condition,'status':'PASS','parameters':counts,'blocked_Query_direct_path_gradient':direct,
                        'unblocked_Query_path_gradient':query_gradient,'gpu_two_batches':grads,
                        'checkpoint_roundtrip':'PASS','constant_external_query_frozen':fixed is not None})
        save_json(HERE/'preflight_progress.json',{'passed':len(results),'total':len(protocol['conditions']),'last':condition['run_id']})
        print('PASS '+condition['run_id'],flush=True)
        del model,opt,out
        torch.cuda.empty_cache()
    # Validate the unchanged ROI implementation at empty and extreme ROI sizes.
    roi_checks=[]
    for name in ['empty','one_sampled_pixel','one_unsampled_pixel','full']:
        logits=torch.randn(2,1,196,device='cuda',requires_grad=True)
        attention=logits.softmax(-1)
        roi=torch.zeros(2,1,224,224,device='cuda')
        if name=='one_sampled_pixel': roi[:,:,0,0]=1
        elif name=='one_unsampled_pixel': roi[:,:,1,1]=1
        elif name=='full': roi[:]=1
        loss,details=roi_kl(attention,roi,(14,14))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert details['valid_roi']==(0 if name in ['empty','one_unsampled_pixel'] else 2)
        roi_checks.append({'case':name,'loss':float(loss.detach()),'gradient_norm':norm([logits.grad]),**details})
    # Same data, RNG and optimization: matched/ON must reproduce original updates.
    matched_training=[]
    for seed in [42,3407,2026]:
        data=next(iter(DataLoader(QueryCohort('train',seed,'Matched'),batch_size=4)))
        set_seed(seed); original=make_model('PGCA-0').cuda().train()
        set_seed(seed); adapted=make_query_model('Matched',True,seed).cuda().train()
        opt1=torch.optim.AdamW(original.parameters(),lr=.0003,weight_decay=.0001)
        opt2=torch.optim.AdamW(adapted.parameters(),lr=.0003,weight_decay=.0001)
        for step in range(2):
            for model,opt in [(original,opt1),(adapted,opt2)]:
                set_seed(seed+step+1)
                opt.zero_grad(set_to_none=True)
                model(**inputs(data))['loss_ord'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
        error=max(float((v-adapted.state_dict()[k]).abs().max()) for k,v in original.state_dict().items())
        assert error==0, error
        matched_training.append({'seed':seed,'max_state_difference_after_two_updates':error})
        del original,adapted,opt1,opt2
    save_json(HERE/'preflight_report.json',{'status':'PASS','time_unix':time.time(),'device':torch.cuda.get_device_name(0),
        'conditions':results,'roi_edge_checks':roi_checks,'matched_historical_update_equivalence':matched_training,
        'empty_downsampled_ROI_behavior':'nearest-neighbor downsampling can remove a single unsampled pixel; skip spatial loss and retain classification, unchanged from frozen implementation'})
    print('ALL_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        save_json(HERE/'preflight_failure.json',{'status':'FAILED','traceback':traceback.format_exc()})
        raise

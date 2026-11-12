"""Frozen-budget Query experiments; selection is validation-only."""
import argparse
import json
import shutil
import subprocess
import time
import traceback
import pandas as pd
from torch.utils.data import DataLoader
from query_model import *
from evaluation import metrics


def evaluate(model, dataset, destination=None):
    model.eval()
    rows, maps = [], []
    with torch.no_grad():
        for data in DataLoader(dataset,batch_size=16,shuffle=False):
            out=model(**inputs(data,with_label=False))
            pred=out['pred'].cpu().numpy()
            probabilities=out['p_ge'].cpu().numpy()
            for j,index in enumerate(data['index'].tolist()):
                r=dataset.rows[index]
                rows.append(dict(image_id=r['image_id'],patient_key=r['patient_key'],group_id=r['group_id'],
                                 y_true=int(data['y'][j]),y_pred=int(pred[j]),
                                 **{f'p_ge_{k}':float(probabilities[j,k-2]) for k in range(2,7)}))
            if destination is not None:
                maps.append(out['attn_map'].reshape(-1,14,14).cpu().numpy())
    frame=pd.DataFrame(rows)
    result=metrics(frame.y_true,frame.y_pred)
    assert result['MAE']+1e-12>=1-result['Accuracy']
    if destination is not None:
        frame.to_csv(destination/'test_predictions.csv',index=False)
        np.save(destination/'test_attention_maps.npy',np.concatenate(maps))
        np.savetxt(destination/'test_confusion_matrix.csv',result['confusion_matrix'],fmt='%d',delimiter=',')
        save_json(destination/'test_metrics.json',result)
    return result


def verify_lock():
    lock=json.loads((HERE/'training_lock.json').read_text())
    for relative,digest in lock['hashes'].items():
        assert sha(HERE/relative)==digest,relative
    return lock


def reuse(condition,model,run):
    historical=BASE/'runs/PGCA-0'/f"seed_{condition['seed']}"
    checkpoint=torch.load(historical/'best_checkpoint.pt',map_location='cuda',weights_only=False)
    config=checkpoint['config']
    frozen=json.loads((HERE/'protocol.json').read_text())['base_config']
    for key in ['epochs','batch_size','learning_rate','weight_decay','gradient_clip_norm']:
        assert config[key]==frozen[key],key
    model.load_state_dict(checkpoint['model'],strict=True)
    test=QueryCohort('test',condition['seed'],'Matched')
    result=evaluate(model,test,run)
    old=pd.read_csv(historical/'test_predictions.csv').sort_values('image_id')
    new=pd.read_csv(run/'test_predictions.csv').sort_values('image_id')
    for key in ['image_id','group_id','y_true','y_pred']:
        assert np.array_equal(old[key].to_numpy(),new[key].to_numpy()),key
    prob_error=float(np.abs(old.filter(like='p_ge_').to_numpy()-new.filter(like='p_ge_').to_numpy()).max())
    map_error=float(np.abs(np.load(historical/'test_attention_maps.npy')-np.load(run/'test_attention_maps.npy')).max())
    assert prob_error<1e-5 and map_error<1e-5,(prob_error,map_error)
    for filename in ['best_checkpoint.pt','training_curve.json']:
        shutil.copy2(historical/filename,run/filename)
        assert sha(historical/filename)==sha(run/filename)
    return dict(status='SUCCESS_REUSED',source=str(historical),best_epoch=checkpoint['epoch'],test=result,
                checkpoint_sha256=sha(run/'best_checkpoint.pt'),probability_max_abs_error=prob_error,
                attention_max_abs_error=map_error,new_training_seconds=0)


def main(run_id):
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    protocol=json.loads((HERE/'protocol.json').read_text())
    condition=next(c for c in protocol['conditions'] if c['run_id']==run_id)
    lock=verify_lock()
    assert json.loads((HERE/'preflight_report.json').read_text())['status']=='PASS'
    assert json.loads((HERE/'overfit_checks.json').read_text())['status']=='PASS'
    run=HERE/'runs'/run_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/'completed.json').exists():
        print('ALREADY_COMPLETE',run_id,flush=True)
        return
    config={**protocol['base_config'],**condition,'training_code_commit':lock['commit'],
            'locked_hashes':lock['hashes'],'selection':'max validation QWK, min validation MAE, earliest epoch',
            'query_mapping_sha256':sha(HERE/'design/query_donors.npy'),'automatic_shutdown':False}
    set_seed(condition['seed'])
    model=make_query_model(condition['query'],condition['residual_on'],condition['seed']).cuda()
    config['parameter_counts']=parameter_counts(model)
    save_json(run/'config.json',config)
    shutil.copy2(HERE/'split_manifest.csv',run/'split_manifest.csv')
    shutil.copy2(HERE/'environment.txt',run/'environment.txt')
    try:
        if condition['reuse_candidate']:
            completed=reuse(condition,model,run)
        else:
            completed=fit(condition,config,model,run)
        completed.update(condition=condition,training_code_commit=lock['commit'])
        save_json(run/'completed.json',completed)
        print('COMPLETED',run_id,completed['status'],flush=True)
    except Exception as error:
        save_json(run/f'failure_{time.time_ns()}.json',dict(condition=condition,error=repr(error),traceback=traceback.format_exc()))
        raise


def fit(condition,config,model,run):
    train=QueryCohort('train',condition['seed'],condition['query'])
    val=QueryCohort('val',condition['seed'],condition['query'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    first,best,best_epoch,curve=1,(-float('inf'),-float('inf')),0,[]
    if (run/'last_checkpoint.pt').exists():
        state=torch.load(run/'last_checkpoint.pt',map_location='cuda',weights_only=False)
        assert state['config']==config
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first,best,best_epoch=state['epoch']+1,tuple(state['selection']),state['best_epoch']
        curve=json.loads((run/'training_curve.json').read_text())
        assert len(curve)==first-1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,config['epochs']+1):
        set_seed(condition['seed']+epoch)
        train.epoch=epoch
        model.train()
        started=time.monotonic()
        sums={'loss':0.,'ordinal':0.,'roi_kl':0.}
        gradient_norms=[]
        ca_norms=[]
        count,valid_roi,mass_error=0,0,0.
        generator=torch.Generator().manual_seed(condition['seed']+epoch)
        for data in DataLoader(train,batch_size=config['batch_size'],shuffle=True,generator=generator):
            optimizer.zero_grad(set_to_none=True)
            out=model(**inputs(data))
            loss,kl,details=objective(out,data['roi'].cuda(),condition['lambda_ROI'])
            assert torch.isfinite(loss), 'nonfinite loss'
            loss.backward()
            parameters=[p for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(p.grad).all() for p in parameters),'nonfinite gradient'
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
            gradient_norms.append(float(norm))
            ca_norms.append(float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.cross_attn.parameters() if p.grad is not None))))
            optimizer.step()
            n=len(data['y']);count+=n
            for key,value in [('loss',loss),('ordinal',out['loss_ord']),('roi_kl',kl)]:
                sums[key]+=float(value.detach())*n
            valid_roi+=details['valid_roi']
            mass_error=max(mass_error,details['alpha_mass_error'],details['mu_mass_error'])
        assert count==2599 and max(ca_norms)>0
        torch.cuda.synchronize()
        training_seconds=time.monotonic()-started
        validation=evaluate(model,val)
        selection=(validation['QWK'],-validation['MAE'])
        if selection>best:
            best,best_epoch=selection,epoch
            save_checkpoint(run/'best_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,config=config,validation=validation))
        record=dict(epoch=epoch,**{k:v/count for k,v in sums.items()},validation=validation,
                    training_seconds=training_seconds,elapsed_seconds=time.monotonic()-started,
                    valid_training_ROI=valid_roi,mass_error=mass_error,
                    gradient_norm_mean=float(np.mean(gradient_norms)),gradient_norm_max=max(gradient_norms),
                    attention_gradient_norm_mean=float(np.mean(ca_norms)),attention_gradient_norm_min=min(ca_norms))
        curve.append(record)
        save_json(run/'training_curve.json',curve)
        save_checkpoint(run/'last_checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                        selection=best,best_epoch=best_epoch,config=config))
        print(json.dumps(dict(run=condition['run_id'],epoch=epoch,loss=record['loss'],val_QWK=validation['QWK'],seconds=record['elapsed_seconds'])),flush=True)
    peak=torch.cuda.max_memory_allocated()
    checkpoint=torch.load(run/'best_checkpoint.pt',map_location='cuda',weights_only=False)
    assert checkpoint['epoch']==best_epoch and len(curve)==30
    model.load_state_dict(checkpoint['model'])
    result=evaluate(model,QueryCohort('test',condition['seed'],condition['query']),run)
    return dict(status='SUCCESS_NEW_FIT',best_epoch=best_epoch,test=result,checkpoint_sha256=sha(run/'best_checkpoint.pt'),
                training_seconds=sum(r['training_seconds'] for r in curve),peak_allocated_bytes=peak,
                full_epoch_seconds=sum(r['elapsed_seconds'] for r in curve),epochs_completed=len(curve))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_id')
    main(parser.parse_args().run_id)

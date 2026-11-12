"""Replay each real selected checkpoint, not a differently named model."""
import json
import hashlib
import pandas as pd
from train_query import evaluate, verify_lock
from query_model import *


def main():
    torch.set_num_threads(8)
    verify_lock()
    records=[]
    for c in json.loads((HERE/'protocol.json').read_text())['conditions']:
        run=HERE/'runs'/c['run_id']
        assert (run/'completed.json').exists()
        set_seed(c['seed'])
        model=make_query_model(c['query'],c['residual_on'],c['seed']).cuda()
        checkpoint=torch.load(run/'best_checkpoint.pt',map_location='cuda',weights_only=False)
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        state_hash=hashlib.sha256()
        for name,value in sorted(checkpoint['model'].items()):
            state_hash.update(name.encode())
            state_hash.update(value.detach().cpu().numpy().tobytes())
        if not c['reuse_candidate']:
            for field in ['run_id','query','residual_on','seed','lambda_ROI']:
                assert checkpoint['config'][field]==c[field]
        model.load_state_dict(checkpoint['model'],strict=True)
        destination=run/'checkpoint_replay';destination.mkdir(exist_ok=True)
        summary=evaluate(model,QueryCohort('test',c['seed'],c['query']),destination)
        a,b=pd.read_csv(run/'test_predictions.csv'),pd.read_csv(destination/'test_predictions.csv')
        for key in ['image_id','group_id','y_true','y_pred']:
            assert np.array_equal(a[key],b[key])
        error=float(np.max(np.abs(a.filter(like='p_ge_').to_numpy()-b.filter(like='p_ge_').to_numpy())))
        map_error=float(np.max(np.abs(np.load(run/'test_attention_maps.npy')-np.load(destination/'test_attention_maps.npy'))))
        assert error<1e-6 and map_error<1e-6,(c,error,map_error)
        records.append(dict(run_id=c['run_id'],probability_max_abs_error=error,map_max_abs_error=map_error,
                            model_state_sha256=state_hash.hexdigest(),status='PASS'))
        print('REPLAY_PASS',c['run_id'],flush=True)
    assert len({r['model_state_sha256'] for r in records})==39,'Unexpected duplicated fitted state'
    save_json(HERE/'analysis/checkpoint_replay.json',dict(status='PASS',distinct_model_states=39,records=records))


if __name__=='__main__': main()

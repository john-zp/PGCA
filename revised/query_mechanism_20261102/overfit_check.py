"""Training-only tiny-cohort optimization check, never a model-selection search."""
import json
from torch.utils.data import DataLoader, Subset
from query_model import *


def main():
    torch.set_num_threads(8)
    conditions=json.loads((HERE/'protocol.json').read_text())['conditions']
    records=[]
    for c in [c for c in conditions if c['seed']==42]:
        set_seed(42)
        model=make_query_model(c['query'],c['residual_on'],42).cuda()
        train=QueryCohort('train',42,c['query'])
        data=next(iter(DataLoader(Subset(train,list(range(50))),batch_size=50)))
        model.eval()
        with torch.no_grad(): initial=float(model(**inputs(data))['loss_ord'])
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        for step in range(80):
            optimizer.zero_grad(set_to_none=True)
            out=model(**inputs(data))
            loss,_,_=objective(out,data['roi'].cuda(),c['lambda_ROI'])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out=model(**inputs(data));final=float(out['loss_ord'])
            accuracy=float((out['pred']==data['y'].cuda()).float().mean())
        assert final<initial*.8,(c,initial,final)
        records.append(dict(condition=c,initial_ordinal_loss=initial,final_ordinal_loss=final,training_accuracy=accuracy,steps=80,n=50))
        print(c['run_id'],initial,final,accuracy,flush=True)
    save_json(HERE/'overfit_checks.json',dict(status='PASS',records=records,scope='diagnostic 50 training samples; checkpoints discarded; not a formal experiment'))


if __name__=='__main__': main()

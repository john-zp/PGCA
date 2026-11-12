"""Finite isolated ROI-control fits; test evaluation is a separate locked step."""
import argparse
import csv
import time
import traceback
from study_common import *
from evaluation import metrics


def fit(family,seed,weight,run):
    lock=verify_lock()
    assert read_json(HERE/"preflight_report.json")["status"]=="PASS"
    config=config_for(family,seed,weight)
    if (run/"training_complete.json").exists():
        old=read_json(run/"config.json")
        assert all(old[k]==config[k] for k in ["model","seed","lambda_ROI","protocol_sha256"])
        return
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"config.json",config)
    save_json(run/"environment.json",{"torch":torch.__version__,"cuda":torch.version.cuda,
              "gpu":torch.cuda.get_device_name(0),"source_lock_sha256":sha(HERE/"source_lock.json")})
    (run/"split_manifest.csv").write_bytes((BASE/"split_manifest.csv").read_bytes())
    train,val=Cohort("train",seed),Cohort("val",seed)
    set_seed(seed)
    model=make_model("PGCA-0").cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    parameters=attention_parameters(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    training_start=time.monotonic()
    best=(-float("inf"),-float("inf"))
    best_epoch=0
    curve=[]
    batch_rows=[]
    component_grads=[]
    for epoch in range(1,31):
        set_seed(seed+epoch)
        train.epoch=epoch
        model.train()
        torch.cuda.synchronize()
        start=time.monotonic()
        truth,preds=[],[]
        epoch_rows=[]
        generator=torch.Generator().manual_seed(seed+epoch)
        for batch_idx,data in enumerate(DataLoader(train,batch_size=16,shuffle=True,generator=generator)):
            opt.zero_grad(set_to_none=True)
            out=model(**inputs(data))
            roi=data["roi"].cuda()
            torch.cuda.synchronize()
            spatial_start=time.monotonic()
            loss,parts,details=objective(out,roi,config)
            torch.cuda.synchronize()
            spatial_seconds=time.monotonic()-spatial_start
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            if batch_idx==0:
                diagnostic={"epoch":epoch,"valid_roi":details["valid_roi"]}
                for key,term in parts.items():
                    grads=torch.autograd.grad(term,[p for _,p in parameters],retain_graph=True,allow_unused=True)
                    assert all(torch.isfinite(g).all() for g in grads if g is not None)
                    diagnostic[key+"_attention_grad_norm"]=norm_of(grads)
                component_grads.append(diagnostic)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite training gradient")
            grad=norm_of([p.grad for p in model.parameters()])
            attention_grad=norm_of([p.grad for _,p in parameters])
            clipped=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
            opt.step()
            truth.extend(data["y"].tolist())
            preds.extend(out["pred"].detach().cpu().tolist())
            row={"epoch":epoch,"batch":batch_idx,"n":len(data["y"]),"total_loss":float(loss.detach()),
                 **{key:float(value.detach()) for key,value in parts.items()},**details,
                 "global_gradient_norm":grad,"attention_gradient_norm":attention_grad,
                 "clip_returned_norm":clipped,"spatial_loss_forward_seconds":spatial_seconds}
            batch_rows.append(row)
            epoch_rows.append(row)
        torch.cuda.synchronize()
        train_seconds=time.monotonic()-start
        val_start=time.monotonic()
        validation=evaluate(model,val,None,"val")
        torch.cuda.synchronize()
        val_seconds=time.monotonic()-val_start
        choice=(validation["QWK"],-validation["MAE"])
        if choice>best:
            best,best_epoch=choice,epoch
            save_checkpoint(run/"best_checkpoint.pt",{"model":model.state_dict(),"epoch":epoch,
                  "config":config,"validation":validation,"source_lock_sha256":sha(HERE/"source_lock.json")})
            save_json(run/"best_validation_metrics.json",validation)
        frame=pd.DataFrame(epoch_rows)
        train_summary=metrics(truth,preds)
        entry={"epoch":epoch,"training":train_summary,"validation":validation,
               "training_loss":float(np.average(frame.total_loss,weights=frame.n)),
               "training_seconds":train_seconds,"validation_seconds":val_seconds,
               "elapsed_seconds":train_seconds+val_seconds,
               "spatial_loss_forward_seconds":float(frame.spatial_loss_forward_seconds.sum()),
               "valid_roi":int(frame.valid_roi.sum()),"maximum_marginal_error":float(frame.marginal_error.max()),
               "mean_roi_kl_valid_cases":float(np.average(frame.roi_kl,weights=frame.valid_roi)),
               "global_gradient_norm_mean":float(frame.global_gradient_norm.mean()),
               "global_gradient_norm_max":float(frame.global_gradient_norm.max()),
               "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        curve.append(entry)
        save_json(run/"training_curve.json",curve)
        pd.DataFrame(batch_rows).to_csv(run/"batch_statistics.csv",index=False)
        pd.DataFrame(component_grads).to_csv(run/"component_gradient_statistics.csv",index=False)
        save_checkpoint(run/"last_checkpoint.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),
                  "epoch":epoch,"best_epoch":best_epoch,"selection":best,"config":config})
        print(json.dumps({"family":family,"seed":seed,"lambda":weight,"epoch":epoch,"val_QWK":validation["QWK"],
                          "val_MAE":validation["MAE"],"seconds":entry["elapsed_seconds"]}),flush=True)
    torch.cuda.synchronize()
    fit_seconds=time.monotonic()-training_start
    checkpoint=torch.load(run/"best_checkpoint.pt",map_location="cuda",weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation=evaluate(model,val,run,"val",with_maps=True)
    assert validation["QWK"]==checkpoint["validation"]["QWK"]
    save_json(run/"training_complete.json",{"status":"COMPLETED","family":family,"seed":seed,"lambda_ROI":weight,
              "best_epoch":best_epoch,"epochs":30,"validation":validation,"training_wall_seconds":fit_seconds,
              "training_epoch_seconds_total":sum(r["training_seconds"] for r in curve),
              "train_plus_val_seconds_total":sum(r["elapsed_seconds"] for r in curve),
              "spatial_loss_forward_seconds_total":sum(r["spatial_loss_forward_seconds"] for r in curve),
              "peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in curve),
              "peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in curve),
              "parameter_count":sum(p.numel() for p in model.parameters()),"checkpoint_sha256":sha(run/"best_checkpoint.pt"),
              "test_not_evaluated":True})


def test_selected(run):
    verify_lock()
    selection=read_json(HERE/"selection_lock.json")
    assert str(run.relative_to(HERE)) in [r["run"] for r in selection["formal_runs"]]
    if (run/"completed.json").exists(): return
    complete=read_json(run/"training_complete.json")
    assert sha(run/"best_checkpoint.pt")==complete["checkpoint_sha256"]
    checkpoint=torch.load(run/"best_checkpoint.pt",map_location="cuda",weights_only=False)
    model=make_model("PGCA-0").cuda()
    model.load_state_dict(checkpoint["model"],strict=True)
    result=evaluate(model,Cohort("test",checkpoint["config"]["seed"]),run,"test",with_maps=True)
    assert result["MAE"]>=1-result["Accuracy"]-1e-12
    save_json(run/"completed.json",{**complete,"test_not_evaluated":False,"test":result,
              "selection_lock_sha256":sha(HERE/"selection_lock.json"),"test_checkpoint_sha256":sha(run/"best_checkpoint.pt")})
    print("TEST_COMPLETE "+str(run),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["fit","test"])
    parser.add_argument("--family",choices=FAMILIES)
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--weight",type=float,choices=WEIGHTS)
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    run=HERE/args.run
    assert run.resolve().is_relative_to(HERE.resolve())
    try:
        if args.mode=="fit": fit(args.family,args.seed,args.weight,run)
        else: test_selected(run)
    except Exception:
        run.mkdir(parents=True,exist_ok=True)
        save_json(run/("failure_"+args.mode+".json"),{"status":"FAILED","traceback":traceback.format_exc()})
        raise

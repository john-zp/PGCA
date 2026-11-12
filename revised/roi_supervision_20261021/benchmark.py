"""Measured no-update cost comparison and checkpoint/prediction replay."""
import gc
import time
from study_common import *
from analyze import selected_runs


def synchronize_time(function,repeats=20,warmup=5):
    for _ in range(warmup): function()
    torch.cuda.synchronize()
    walls,devices=[],[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        before=time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter()-before)*1000)
        devices.append(start.elapsed_time(end))
    return {"wall_ms_median":float(np.median(walls)),"wall_ms_mean":float(np.mean(walls)),
            "cuda_ms_median":float(np.median(devices)),"cuda_ms_mean":float(np.mean(devices)),
            "repeats":repeats,"warmup":warmup}


def main():
    torch.set_num_threads(8)
    verify_lock()
    assert read_json(HERE/"run_status.json")["phase"]=="TRAINING_COMPLETE"
    runs=selected_runs()
    batch=next(iter(DataLoader(Cohort("train",42),batch_size=16,shuffle=False)))
    data=inputs(batch)
    roi=batch["roi"].cuda()
    output=[]
    replays=[]
    for (name,seed),run in runs.items():
        set_seed(seed)
        checkpoint=torch.load(run/"best_checkpoint.pt",map_location="cuda",weights_only=False)
        model=make_model("PGCA-0").cuda()
        model.load_state_dict(checkpoint["model"],strict=True)
        model.eval()
        expected=pd.read_csv(run/"test_predictions.csv")
        native=np.load(run/"test_attention_maps.npy")
        reproduced,maps=[],[]
        with torch.no_grad():
            for item in DataLoader(Cohort("test",seed),batch_size=16,shuffle=False):
                x=inputs(item)
                x.pop("y")
                pred=model(**x)
                reproduced.extend(pred["pred"].cpu().tolist())
                maps.append(pred["attn_map"].cpu().numpy().reshape(-1,14,14))
        assert np.array_equal(expected.y_pred.to_numpy(),reproduced)
        delta=float(np.abs(np.concatenate(maps)-native).max())
        assert delta<1e-6,(name,seed,delta)
        replays.append({"model":name,"seed":seed,"predictions_exact":True,"map_max_absolute_difference":delta,
                        "checkpoint_sha256":sha(run/"best_checkpoint.pt")})
        for bs in [1,16]:
            x={k:v[:bs] for k,v in data.items() if k!="y"}
            with torch.no_grad():
                values=synchronize_time(lambda:model(**x),repeats=50,warmup=10)
            output.append({"model":name,"seed":seed,"task":"inference_no_ROI","batch_size":bs,
                           "parameters":sum(p.numel() for p in model.parameters()),**values})
        model.train()
        set_seed(42)
        out=model(**data)
        attention=out["attn_map"].detach().clone().requires_grad_(True)
        config=checkpoint["config"]
        def spatial():
            attention.grad=None
            total=attention.sum()*0
            if name in FAMILIES:
                kl,_=roi_kl(attention,roi,(14,14))
                total=total+config["lambda_ROI"]*kl
            if name in ["PGCA-staticOT","PGCA-full-SBpath",FAMILIES[1]]:
                ot,path,_=alignment(attention,roi,(14,14),name=="PGCA-full-SBpath",.05,200,.3)
                total=total+.2*ot
                if name=="PGCA-full-SBpath": total=total+.05*path
            total.backward()
        torch.cuda.reset_peak_memory_stats()
        values=synchronize_time(spatial,repeats=10,warmup=2)
        output.append({"model":name,"seed":seed,"task":"isolated_spatial_loss_forward_backward",
                       "batch_size":16,"valid_roi":int((torch.nn.functional.interpolate(roi,size=(14,14),mode="nearest").flatten(1).sum(1)>0).sum()),
                       "peak_allocated_bytes":torch.cuda.max_memory_allocated(),**values})
        del out,attention,model,checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print("BENCHMARK "+name+" "+str(seed),flush=True)
    pd.DataFrame(output).to_csv(HERE/"cost_microbenchmark.csv",index=False)
    pd.DataFrame(replays).to_csv(HERE/"checkpoint_replay.csv",index=False)
    save_json(HERE/"benchmark_status.json",{"status":"PASS","gpu":torch.cuda.get_device_name(0),
              "torch":torch.__version__,"cuda":torch.version.cuda,"no_optimizer_steps":True,
              "benchmark_source_sha256":sha(HERE/"benchmark.py"),"deterministic_flags_reset_before_each_model":True,
              "training_batch_indices":batch["index"].tolist(),"checkpoint_replays":len(replays),
              "parameter_count_equal":len({r["parameters"] for r in output if "parameters" in r})==1})
    verify_lock()
    print("BENCHMARK_COMPLETE",flush=True)


if __name__=="__main__": main()

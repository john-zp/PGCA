"""Finite numerical and provenance gates before any candidate fitting."""
import math
import time
import traceback
from study_common import *


def numerical_tests():
    cases=[]
    for pixels in [1,4,49,196]:
        logits=torch.zeros(2,1,196,device="cuda",requires_grad=True)
        attn=logits.softmax(-1)
        mask=torch.zeros(2,1,14,14,device="cuda")
        mask.flatten(1)[:,:pixels]=1
        loss,details=roi_kl(attn,mask,(14,14))
        assert abs(float(loss)-math.log(196/pixels))<2e-6
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        cases.append({"roi_tokens":pixels,"KL":float(loss.detach()),"logit_gradient_norm":norm_of([logits.grad]),**details})
    attn=torch.full((2,1,196),1/196,device="cuda",requires_grad=True)
    empty=torch.zeros(2,1,224,224,device="cuda")
    loss,details=roi_kl(attn,empty,(14,14))
    loss.backward()
    assert float(loss)==0 and details["valid_roi"]==0 and torch.isfinite(attn.grad).all()
    tiny=empty.clone()
    tiny[0,0,0,0]=1
    tiny[1,0,1,1]=1
    loss,details=roi_kl(attn,tiny,(14,14))
    assert details["valid_roi"]==1 and torch.isfinite(loss)
    zeros=torch.zeros(1,1,196,device="cuda",requires_grad=True)
    zeros.data[0,0,-1]=1
    roi=torch.zeros(1,1,14,14,device="cuda")
    roi[0,0,0,0]=1
    extreme,extreme_details=roi_kl(zeros,roi,(14,14))
    extreme.backward()
    assert torch.isfinite(zeros.grad).all()
    return {"area_tests":cases,"empty_roi":"PASS","tiny_roi_valid_count":details["valid_roi"],
            "zero_attention_entry_KL":float(extreme.detach()),"zero_entry_gradient_finite":True}


def overfit(family,train):
    set_seed(42)
    model=make_model("PGCA-0").cuda().train()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
        if isinstance(module,torch.nn.MultiheadAttention): module.dropout=0.
    chosen=np.random.default_rng(42).choice(len(train),50,replace=False)
    subset=torch.utils.data.Subset(train,chosen.tolist())
    batch=next(iter(DataLoader(subset,batch_size=50)))
    data=inputs(batch)
    roi=batch["roi"].cuda()
    config=config_for(family,42,.05)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    records=[]
    start=time.monotonic()
    for step in range(201):
        opt.zero_grad(set_to_none=True)
        out=model(**data)
        loss,parts,_=objective(out,roi,config)
        assert torch.isfinite(loss)
        if step%25==0:
            acc=float((out["pred"]==data["y"]).float().mean())
            row={"step":step,"accuracy":acc,"ordinal_loss":float(parts["ordinal"].detach()),"total_loss":float(loss.detach())}
            records.append(row)
            print(json.dumps({"overfit":family,**row}),flush=True)
            if step>0 and acc>=.9 and row["ordinal_loss"]<records[0]["ordinal_loss"]*.4:
                break
        if step==200:
            raise RuntimeError("Small-sample overfit gate not met for "+family)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
    return {"status":"PASS","sample_count":50,"train_item_indices":chosen.tolist(),"curve":records,
            "elapsed_seconds":time.monotonic()-start,"diagnostic_lr":.003,"diagnostic_dropout":0,
            "not_formal_training":True}


def main():
    torch.set_num_threads(8)
    assert torch.cuda.is_available(),"CUDA required"
    old=read_json(BASE/"source_lock.json")
    hashes={}
    for category in ["code_hashes","dataset_hashes","dependency_hashes","source_data_hashes"]:
        for name,expected in old[category].items():
            path=Path(name) if Path(name).is_absolute() else BASE/name
            actual=sha(path)
            assert actual==expected,(category,str(path))
            hashes[str(path)]=actual
    manifest=pd.read_csv(BASE/"data/dataset_manifest.csv")
    assert manifest.split.value_counts().to_dict()=={"train":2599,"val":555,"test":555}
    assert manifest.image_id.is_unique
    assert manifest.groupby("group_id").split.nunique().max()==1
    assert manifest.groupby("patient_key").split.nunique().max()==1
    features=np.load(BASE/"data/features36.npy",mmap_mode="r")
    embeddings=np.load(BASE/"data/pathology_embeddings.npy",mmap_mode="r")
    labels=np.load(BASE/"data/y.npy",mmap_mode="r")
    assert features.shape==(3709,36) and np.isin(features,[0,1]).all()
    assert embeddings.shape==(3709,512) and np.isfinite(embeddings).all()
    assert np.isin(labels,range(1,7)).all()
    masks=np.load(BASE/"data/roi_masks.npy",mmap_mode="r")
    test=manifest[manifest.split=="test"]
    valid=np.asarray(masks[test.index]).reshape(555,-1).sum(1)>0
    assert valid.sum()==383 and test.loc[valid,"group_id"].nunique()==290
    subset=pd.read_csv(BASE/"mechanism_validation_20261014/localization_roi_subset.csv")
    assert set(test.loc[valid,"image_id"])==set(subset.image_id)
    checkpoint_audit={r["checkpoint"]:r["sha256"] for r in read_json(BASE/"checkpoint_integrity_audit.json")["checkpoints"]}
    for name in HISTORICAL:
        for seed in SEEDS:
            run=BASE/"runs"/name/f"seed_{seed}"
            for filename in ["best_checkpoint.pt","test_predictions.csv","test_attention_maps.npy","training_curve.json","config.json","completed.json"]:
                path=run/filename
                hashes[str(path)]=sha(path)
            assert hashes[str(run/"best_checkpoint.pt")]==checkpoint_audit[str((run/"best_checkpoint.pt").relative_to(BASE))]
    env=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).strip()+"\n"
    import hashlib
    env_sha=hashlib.sha256(env.encode()).hexdigest()
    assert env_sha==old["environment_sha256"],"Historical dependency environment differs"
    (HERE/"environment.txt").write_text(env)
    (HERE/"split_manifest.csv").write_bytes((BASE/"split_manifest.csv").read_bytes())
    train=Cohort("train",42)
    numerical=numerical_tests()
    smoke={}
    for family in FAMILIES:
        set_seed(42)
        model=make_model("PGCA-0").cuda().train()
        config=config_for(family,42,.05)
        parameters=attention_parameters(model)
        assert parameters
        records=[]
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        for index,data in enumerate(DataLoader(train,batch_size=16,shuffle=False)):
            opt.zero_grad(set_to_none=True)
            out=model(**inputs(data))
            assert out["attn_map"].shape==(16,1,196) and out["fusion_out"].shape==(16,256)
            assert out["logits"].shape==(16,5)
            loss,parts,details=objective(out,data["roi"].cuda(),config)
            grad=torch.autograd.grad(parts["roi_kl"],[p for _,p in parameters],retain_graph=True,allow_unused=True)
            gn=norm_of(grad)
            assert gn>0 and np.isfinite(gn)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            records.append({"batch":index,"total":float(loss.detach()),"roi_kl_grad_norm":gn,
                            "all_grad_norm":norm_of([p.grad for p in model.parameters()]),**details})
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            if index==1: break
        model.eval()
        no_label=inputs(data)
        no_label.pop("y")
        with torch.no_grad():
            pred=model(**no_label)
            assert pred["pred"].shape==(16,)
        smoke[family]={"status":"PASS","parameter_count":sum(p.numel() for p in model.parameters()),
                       "fusion_input_dim":model.fusion[0].in_features,"attention_parameter_names":[n for n,_ in parameters],
                       "batches":records,"inference_without_roi":True}
    assert smoke[FAMILIES[0]]["parameter_count"]==smoke[FAMILIES[1]]["parameter_count"]
    save_json(HERE/"preflight_progress.json",{"numerical":numerical,"smoke":smoke})
    overfits={family:overfit(family,train) for family in FAMILIES}
    code=["study_common.py","preflight.py","train_roi.py","workflow.py","protocol.md"]
    lock={"read_only_hashes":hashes,"training_code_hashes":{p:sha(HERE/p) for p in code},
          "code_commit":git_commit(),"historical_code_commit":git_commit(BASE),
          "environment_sha256":env_sha,"gpu":torch.cuda.get_device_name(0),
          "torch":torch.__version__,"cuda":torch.version.cuda,"protocol_sha256":sha(HERE/"protocol.md")}
    save_json(HERE/"source_lock.json",lock)
    save_json(HERE/"preflight_report.json",{"status":"PASS","counts":manifest.split.value_counts().to_dict(),
        "patient_directory_count":int(manifest.patient_key.nunique()),"risk_groups":int(manifest.group_id.nunique()),
        "numerical":numerical,"smoke":smoke,"overfit50":overfits,"gpu":lock["gpu"],"historical_hashes_verified":len(hashes)})
    print("PREFLIGHT_PASS",flush=True)


if __name__=="__main__":
    try: main()
    except Exception:
        save_json(HERE/"preflight_failure.json",{"status":"FAILED","traceback":traceback.format_exc()})
        raise

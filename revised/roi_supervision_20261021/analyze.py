"""Recompute outcomes and paired exploratory statistics from frozen outputs."""
import math
from study_common import *
sys.path.insert(0,str(SUP))
from statistics_common import hard_metrics, classification_pair, scalar_pair, with_mean_sd, METRICS, CLASSES
from localization_area import overlap
from localization_tie_sensitivity import bounds

COMPARISONS=[("PGCA-full-SBpath","PGCA-ROI-KL"),
             ("PGCA-full-SBpath","PGCA-staticOT-ROI-KL"),
             ("PGCA-staticOT-ROI-KL","PGCA-staticOT")]
FRACTIONS=[.01,.025,.05,.075,.1,.15,.2]


def selected_runs():
    runs={(name,seed):BASE/"runs"/name/f"seed_{seed}" for name in HISTORICAL for seed in SEEDS}
    selection=read_json(HERE/"selection_lock.json")
    for row in selection["formal_runs"]:
        directory=HERE/row["run"]
        if (directory/"completed.json").exists(): runs[row["family"],row["seed"]]=directory
    return runs


def classifications(runs,test):
    results,classes,integrity=[],[],[]
    predictions={name:{} for name in HISTORICAL+FAMILIES}
    for (model,seed),run in runs.items():
        frame=pd.read_csv(run/"test_predictions.csv")
        assert len(frame)==555 and np.array_equal(frame.image_id,test.image_id)
        assert np.array_equal(frame.group_id,test.group_id)
        expected=np.load(BASE/"data/y.npy",mmap_mode="r")[test.global_index]
        assert np.array_equal(frame.y_true,expected)
        values=hard_metrics(frame)
        stored=read_json(run/"test_metrics.json")
        assert all(abs(stored[k]-values[k])<1e-12 for k in METRICS)
        cm=np.bincount((frame.y_true.to_numpy(int)-1)*6+frame.y_pred.to_numpy(int)-1,minlength=36).reshape(6,6)
        assert np.array_equal(cm,stored["confusion_matrix"])
        diagonal=cm.diagonal().astype(float)
        precision=np.divide(diagonal,cm.sum(0),out=np.zeros(6),where=cm.sum(0)>0)
        recall=np.divide(diagonal,cm.sum(1),out=np.zeros(6),where=cm.sum(1)>0)
        context={"model":model,"seed":seed,"status":"COMPLETED","test_n":555}
        for j,grade in enumerate(CLASSES):
            values["Precision_"+grade]=float(precision[j])
            values["Recall_"+grade]=float(recall[j])
            classes.append({**context,"BI-RADS":grade,"Precision":precision[j],"Recall":recall[j],"F1":values["F1_"+grade],"support":int(cm.sum(1)[j])})
        results.append({**context,**values})
        predictions[model][seed]=frame
        (HERE/"confusion_matrices").mkdir(exist_ok=True)
        np.savetxt(HERE/"confusion_matrices"/f"{model}_seed_{seed}.csv",cm,fmt="%d",delimiter=",")
        complete=read_json(run/"completed.json")
        curve=read_json(run/"training_curve.json")
        chosen=max(curve,key=lambda x:(x["validation"]["QWK"],-x["validation"]["MAE"],-x["epoch"]))
        assert chosen["epoch"]==complete["best_epoch"]
        if model in FAMILIES:
            assert len(curve)==30 and [r["epoch"] for r in curve]==list(range(1,31))
            assert complete["test_checkpoint_sha256"]==sha(run/"best_checkpoint.pt")
            assert complete["selection_lock_sha256"]==sha(HERE/"selection_lock.json")
            batches=pd.read_csv(run/"batch_statistics.csv")
            assert np.isfinite(batches.select_dtypes("number").to_numpy()).all()
            assert batches.groupby("epoch").n.sum().eq(2599).all()
        integrity.append({**context,"best_epoch":complete["best_epoch"],"checkpoint_sha256":sha(run/"best_checkpoint.pt"),
                          "prediction_sha256":sha(run/"test_predictions.csv"),"attention_sha256":sha(run/"test_attention_maps.npy"),
                          "metrics_recomputed":True,"validation_selection_verified":True,"MAE_accuracy_consistent":True})
    data=pd.DataFrame(results)
    columns=METRICS+[f"{metric}_{grade}" for metric in ["Precision","Recall","F1"] for grade in CLASSES]
    with_mean_sd(data,["model"],columns).to_csv(HERE/"simple_roi_baseline_results.csv",index=False)
    pd.DataFrame(classes).to_csv(HERE/"class_metrics.csv",index=False)
    pd.DataFrame(integrity).to_csv(HERE/"run_integrity.csv",index=False)
    paired=[]
    for a,b in COMPARISONS:
        if set(predictions[a])==set(SEEDS)==set(predictions[b]):
            paired.extend(classification_pair(predictions[a],predictions[b],a+" minus "+b))
    pd.DataFrame(paired).to_csv(HERE/"classification_paired.csv",index=False)
    return data


def localization(runs,test):
    masks=np.load(BASE/"data/roi_masks.npy",mmap_mode="r")[test.global_index][:,0]>0
    valid=masks.reshape(555,-1).any(1)
    assert valid.sum()==383 and test.loc[valid,"group_id"].nunique()==290
    records,tied=[] ,[]
    for (model,seed),run in runs.items():
        native=np.load(run/"test_attention_maps.npy")
        assert native.shape==(555,14,14) and np.isfinite(native).all()
        large=torch.nn.functional.interpolate(torch.from_numpy(native[:,None]),size=(224,224),mode="bilinear",align_corners=False).numpy()[:,0]
        for i in np.flatnonzero(valid):
            heat,roi=large[i],masks[i]
            normalized=(heat-heat.min())/max(float(heat.max()-heat.min()),1e-12)
            context={"model":model,"seed":seed,"image_id":test.iloc[i].image_id,"group_id":test.iloc[i].group_id,
                     "Pointing_Game":float(roi.ravel()[np.argmax(heat)])}
            for level in np.arange(1,10)/10:
                records.append({**context,"method":"threshold","level":float(level),"boundary_tie":False,**overlap(normalized>=level,roi)})
            order=np.argsort(-heat.ravel(),kind="stable")
            for fraction in FRACTIONS:
                k=math.ceil(fraction*roi.size)
                selected=np.zeros(roi.size,dtype=bool)
                selected[order[:k]]=True
                values=overlap(selected.reshape(roi.shape),roi)
                bound=bounds(heat,roi,k)
                assert bound["IoU_lower"]-1e-12<=values["IoU"]<=bound["IoU_upper"]+1e-12
                records.append({**context,"method":"top_k_fraction","level":fraction,
                                "boundary_tie":bool(heat.ravel()[order[k-1]]==heat.ravel()[order[k]]),**values})
                tied.append({**context,"level":fraction,**bound})
        print("LOCALIZATION "+model+" "+str(seed),flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/"localization_per_image.csv",index=False)
    bound_frame=pd.DataFrame(tied)
    bound_frame.to_csv(HERE/"topk_tie_bounds_per_image.csv",index=False)
    columns=["Pointing_Game","IoU","ROI_coverage","precision","predicted_area_fraction","ROI_area_fraction","boundary_tie"]
    summary=frame.groupby(["model","seed","method","level"])[columns].mean().reset_index()
    summary["ROI_n"],summary["ROI_groups"]=383,290
    result=with_mean_sd(summary,["model","method","level"],columns)
    result["ROI_n"],result["ROI_groups"]=383,290
    result.to_csv(HERE/"localization_comparison.csv",index=False)
    historical=pd.read_csv(SUP/"localization_area_per_image.csv")
    joined=frame[frame.model.isin(HISTORICAL)].merge(historical,on=["model","seed","method","level","image_id","group_id"],suffixes=("_new","_old"),validate="one_to_one")
    assert len(joined)==len(historical)
    for col in columns:
        assert np.allclose(joined[col+"_new"].astype(float),joined[col+"_old"].astype(float),atol=1e-12,rtol=0),col
    comparisons,sensitivity=[],[]
    for a,b in COMPARISONS:
        if not all((name,seed) in runs for name in [a,b] for seed in SEEDS): continue
        for method,level in frame[["method","level"]].drop_duplicates().itertuples(index=False,name=None):
            left=frame[(frame.model==a)&(frame.method==method)&(frame.level==level)]
            right=frame[(frame.model==b)&(frame.method==method)&(frame.level==level)]
            for value in ["IoU","ROI_coverage","predicted_area_fraction"]:
                comparisons.extend(scalar_pair(left,right,value,f"{a} minus {b}; {method}={level:g}"))
        left=frame[(frame.model==a)&(frame.method=="threshold")&(frame.level==.5)]
        right=frame[(frame.model==b)&(frame.method=="threshold")&(frame.level==.5)]
        comparisons.extend(scalar_pair(left,right,"Pointing_Game",a+" minus "+b))
        for fraction in FRACTIONS:
            left=bound_frame[(bound_frame.model==a)&(bound_frame.level==fraction)].copy()
            right=bound_frame[(bound_frame.model==b)&(bound_frame.level==fraction)].copy()
            for label,lcol,rcol in [("pessimistic","IoU_lower","IoU_upper"),("optimistic","IoU_upper","IoU_lower")]:
                left["IoU"],right["IoU"]=left[lcol],right[rcol]
                sensitivity.extend(scalar_pair(left,right,"IoU",f"{a} minus {b}; {label}; top_k_fraction={fraction:g}"))
    pd.DataFrame(comparisons).to_csv(HERE/"localization_paired.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(HERE/"localization_tie_sensitivity.csv",index=False)
    save_json(HERE/"localization_integrity.json",{"status":"PASS","ROI_n":383,"ROI_groups":290,
            "historical_replayed_rows":len(joined),"historical_replay_max_tolerance":1e-12,
            "stable_row_major_ties":True,"main_topk_within_bounds":True})


def main():
    torch.set_num_threads(8)
    verify_lock()
    assert read_json(HERE/"run_status.json")["phase"]=="TRAINING_COMPLETE"
    test=pd.read_csv(BASE/"data/dataset_manifest.csv")
    test=test[test.split=="test"].reset_index(names="global_index")
    runs=selected_runs()
    classifications(runs,test)
    localization(runs,test)
    save_json(HERE/"analysis_status.json",{"status":"PASS","successful_runs":len(runs),"complete_five_by_three":len(runs)==15})
    print("ANALYSIS_COMPLETE",flush=True)


if __name__=="__main__": main()

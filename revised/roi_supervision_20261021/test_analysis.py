"""CPU-only regression tests for reused statistics and localization conventions."""
import math
from study_common import *
sys.path.insert(0,str(SUP))
from statistics_common import hard_metrics, classification_pair, scalar_pair
from localization_area import overlap
from localization_tie_sensitivity import bounds
from evaluation import metrics


def main():
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({"image_id":[f"synthetic_{i}" for i in range(60)],
        "group_id":[f"group_{i//2}" for i in range(60)],
        "y_true":np.tile(np.arange(1,7),10),"y_pred":rng.integers(1,7,60)})
    expected=metrics(frame.y_true,frame.y_pred)
    calculated=hard_metrics(frame)
    assert all(abs(expected[m]-calculated[m])<1e-12 for m in ["Accuracy","MAE","QWK","Macro-F1"])
    results=classification_pair({s:frame for s in SEEDS},{s:frame for s in SEEDS},"identical",replicates=50)
    assert all(r["difference"]==r["ci_low"]==r["ci_high"]==0 for r in results)
    left=pd.concat([frame.assign(seed=s,value=np.arange(60)/60) for s in SEEDS],ignore_index=True)
    right=left.copy()
    right["value"]=right.value-.25
    paired=scalar_pair(left,right,"value","constant_difference",replicates=50)
    assert all(abs(r[k]-.25)<1e-12 for r in paired for k in ["difference","ci_low","ci_high"])
    heat=np.ones((4,4),dtype=np.float32)
    roi=np.zeros((4,4),dtype=bool)
    roi[0,:]=True
    selected=np.zeros(16,dtype=bool)
    selected[np.argsort(-heat.ravel(),kind="stable")[:4]]=True
    point=overlap(selected.reshape(4,4),roi)
    bound=bounds(heat,roi,4)
    assert point["IoU"]==1 and bound["IoU_lower"]==0 and bound["IoU_upper"]==1
    assert selected.sum()==math.ceil(.25*roi.size)
    save_json(HERE/"analysis_tests.json",{"status":"PASS","hard_metric_reference":"sklearn frozen implementation",
        "identical_paired_comparison_zero":True,"cluster_scalar_constant_difference":True,
        "stable_topk_and_tie_bounds":True,"no_training":True})
    print("ANALYSIS_TESTS_PASS")


if __name__=="__main__": main()

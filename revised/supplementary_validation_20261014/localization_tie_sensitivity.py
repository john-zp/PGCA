"""Post-hoc numerical audit of tied Top-k boundaries, not a new tuned metric."""
import math

from query_model import BASE, HERE
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from statistics_common import scalar_pair
from train_one import save_json, sha


def bounds(heat, roi, k):
    scores, truth = heat.ravel(), roi.ravel()
    cutoff = np.partition(scores, len(scores)-k)[len(scores)-k]
    above, tied = scores > cutoff, scores == cutoff
    slots = k - int(above.sum())
    positives = int(truth[tied].sum())
    negatives = int(tied.sum()) - positives
    guaranteed = int(truth[above].sum())
    lo = guaranteed + max(0, slots-negatives)
    hi = guaranteed + min(slots, positives)
    area = int(truth.sum())
    return {"IoU_lower": lo/(k+area-lo), "IoU_upper": hi/(k+area-hi),
            "coverage_lower": lo/area, "coverage_upper": hi/area,
            "boundary_tied_pixels": int(tied.sum()), "boundary_slots": slots,
            "ambiguous_boundary": 0 < slots < int(tied.sum())}


def main():
    torch.set_num_threads(8)
    manifest = pd.read_csv(BASE / "data/dataset_manifest.csv")
    test = manifest[manifest.split == "test"].reset_index(names="global_index")
    masks = np.load(BASE / "data/roi_masks.npy", mmap_mode="r")[test.global_index][:, 0] > 0
    valid = masks.reshape(555, -1).any(1)
    assert valid.sum() == 383
    rows = []
    for model in ["PGCA-staticOT", "PGCA-full-SBpath"]:
        for seed in [42, 3407, 2026]:
            directory = BASE / "runs" / model / f"seed_{seed}"
            pred = pd.read_csv(directory / "test_predictions.csv")
            assert np.array_equal(pred.image_id, test.image_id)
            native = np.load(directory / "test_attention_maps.npy")
            large = F.interpolate(torch.from_numpy(native[:, None]), size=(224, 224), mode="bilinear", align_corners=False).numpy()[:, 0]
            for i in np.flatnonzero(valid):
                for fraction in [.01, .025, .05, .075, .1, .15, .2]:
                    rows.append({"model": model, "seed": seed, "image_id": test.iloc[i].image_id,
                        "group_id": test.iloc[i].group_id, "level": fraction,
                        **bounds(large[i], masks[i], math.ceil(fraction*224*224))})
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "localization_topk_tie_bounds.csv", index=False)
    summary = frame.groupby(["model", "seed", "level"])[["IoU_lower", "IoU_upper", "coverage_lower", "coverage_upper", "ambiguous_boundary"]].mean().reset_index()
    summary.to_csv(HERE / "localization_tie_sensitivity_summary.csv", index=False)
    existing = pd.read_csv(HERE / "localization_area_per_image.csv")
    existing = existing[(existing.method == "top_k_fraction") & existing.model.isin(["PGCA-staticOT", "PGCA-full-SBpath"])]
    checked = frame.merge(existing[["model", "seed", "image_id", "level", "IoU"]], on=["model", "seed", "image_id", "level"], validate="one_to_one")
    assert np.all(checked.IoU >= checked.IoU_lower-1e-12)
    assert np.all(checked.IoU <= checked.IoU_upper+1e-12)
    comparisons = []
    for fraction in [.01, .025, .05, .075, .1, .15, .2]:
        left = frame[(frame.model == "PGCA-full-SBpath") & (frame.level == fraction)].copy()
        right = frame[(frame.model == "PGCA-staticOT") & (frame.level == fraction)].copy()
        for name, col_left, col_right in [("pessimistic_tie_margin", "IoU_lower", "IoU_upper"), ("optimistic_tie_margin", "IoU_upper", "IoU_lower")]:
            left["IoU"], right["IoU"] = left[col_left], right[col_right]
            comparisons.extend(scalar_pair(left, right, "IoU", f"Full minus StaticOT; {name}; top_k_fraction={fraction:g}"))
    pd.DataFrame(comparisons).to_csv(HERE / "localization_tie_bounds_paired.csv", index=False)
    save_json(HERE / "localization_tie_audit.json", {"status": "PASS", "reason": "Post-hoc numerical sensitivity after discovering exact tied Top-k boundaries",
        "note": "Bounds use ROI membership only to bound every admissible tie choice; they are not an ROI-selected prediction method or deployable performance estimate.",
        "main_topk_rule_unchanged": "stable_row_major", "new_training": False, "new_inference": False,
        "source_sha256": sha(__import__("pathlib").Path(__file__)), "main_topk_values_within_bounds": True})
    print("TIE_BOUNDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

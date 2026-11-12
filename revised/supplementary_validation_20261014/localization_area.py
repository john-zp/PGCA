"""Area-controlled analysis of frozen maps only; no model loading/inference."""
import json
import math
from pathlib import Path

from query_model import BASE, HERE
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from train_one import sha, save_json
from statistics_common import scalar_pair


def overlap(selected, roi):
    intersection = np.count_nonzero(selected & roi)
    area, roi_area = np.count_nonzero(selected), np.count_nonzero(roi)
    return {"IoU": intersection/(area+roi_area-intersection), "ROI_coverage": intersection/roi_area,
            "precision": intersection/area if area else 0., "predicted_area_fraction": area/roi.size,
            "ROI_area_fraction": roi_area/roi.size}


def main():
    torch.set_num_threads(8)
    receipt = HERE / "localization_area_integrity.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("analysis_sha256") == sha(Path(__file__)):
            assert all(sha(BASE / name) == digest for name, digest in previous["saved_map_hashes"].items())
            assert all(sha(HERE / name) == digest for name, digest in previous["output_hashes"].items())
            print("LOCALIZATION_AREA_ALREADY_COMPLETE_VERIFIED", flush=True)
            return
    manifest = pd.read_csv(BASE / "data/dataset_manifest.csv")
    test = manifest[manifest.split == "test"].reset_index(names="global_index")
    masks = np.load(BASE / "data/roi_masks.npy", mmap_mode="r")[test.global_index.to_numpy()][:, 0] > 0
    valid = masks.reshape(555, -1).any(1)
    assert valid.sum() == 383
    verified = pd.read_csv(BASE / "mechanism_validation_20261014/localization_roi_subset.csv")
    assert set(test.loc[valid, "image_id"]) == set(verified.image_id)
    records, point_records, source_hashes = [], [], {}
    for model in ["PGCA-0", "PGCA-staticOT", "PGCA-full-SBpath"]:
        for seed in [42, 3407, 2026]:
            directory = BASE / "runs" / model / f"seed_{seed}"
            pred = pd.read_csv(directory / "test_predictions.csv")
            assert np.array_equal(pred.image_id, test.image_id)
            native = np.load(directory / "test_attention_maps.npy")
            assert native.shape == (555, 14, 14) and np.isfinite(native).all()
            source_hashes[str((directory / "test_attention_maps.npy").relative_to(BASE))] = sha(directory / "test_attention_maps.npy")
            large = F.interpolate(torch.from_numpy(native[:, None]), size=(224, 224), mode="bilinear", align_corners=False).numpy()[:, 0]
            for i in np.flatnonzero(valid):
                heat, roi = large[i], masks[i]
                normalized = (heat-heat.min())/max(float(heat.max()-heat.min()), 1e-12)
                hit = float(roi.ravel()[np.argmax(heat)])
                context = {"model": model, "seed": seed, "image_id": test.iloc[i].image_id, "group_id": test.iloc[i].group_id, "Pointing_Game": hit}
                point_records.append(context)
                for threshold in np.arange(1, 10)/10:
                    records.append({**context, "method": "threshold", "level": float(threshold), "boundary_tie": False, **overlap(normalized >= threshold, roi)})
                order = np.argsort(-heat.ravel(), kind="stable")
                for fraction in [.01, .025, .05, .075, .1, .15, .2]:
                    k = math.ceil(fraction*roi.size)
                    selected = np.zeros(roi.size, dtype=bool)
                    selected[order[:k]] = True
                    tie = bool(heat.ravel()[order[k-1]] == heat.ravel()[order[k]])
                    records.append({**context, "method": "top_k_fraction", "level": fraction, "boundary_tie": tie, **overlap(selected.reshape(roi.shape), roi)})
    data = pd.DataFrame(records)
    data.to_csv(HERE / "localization_area_per_image.csv", index=False)
    value_columns = ["IoU", "ROI_coverage", "precision", "predicted_area_fraction", "ROI_area_fraction", "Pointing_Game", "boundary_tie"]
    summary = data.groupby(["model", "seed", "method", "level"])[value_columns].mean().reset_index()
    summary["ROI_n"] = 383
    summary["ROI_groups"] = test.loc[valid, "group_id"].nunique()
    aggregate = summary.groupby(["model", "method", "level"])[value_columns].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    summary.to_csv(HERE / "localization_area_control.csv", index=False)
    aggregate.reset_index().to_csv(HERE / "localization_area_control_summary.csv", index=False)
    comparisons = []
    for method, level in data[["method", "level"]].drop_duplicates().itertuples(index=False, name=None):
        part = data[(data.method == method) & (data.level == level)]
        for other in ["PGCA-staticOT", "PGCA-0"]:
            for metric in ["IoU", "ROI_coverage"]:
                comparisons.extend(scalar_pair(part[part.model == "PGCA-full-SBpath"], part[part.model == other], metric,
                    f"Full minus {other}; {method}={level:g}"))
    points = pd.DataFrame(point_records)
    for other in ["PGCA-staticOT", "PGCA-0"]:
        comparisons.extend(scalar_pair(points[points.model == "PGCA-full-SBpath"], points[points.model == other], "Pointing_Game", "Full minus " + other))
    pd.DataFrame(comparisons).to_csv(HERE / "localization_paired_comparisons.csv", index=False)
    for relative, digest in source_hashes.items():
        assert sha(BASE / relative) == digest
    original = pd.read_csv(BASE / "mechanism_validation_20261014/localization_per_image_rechecked.csv")
    reference = data[(data.method == "threshold") & (data.level == .5)].merge(original,
        on=["model", "seed", "image_id", "group_id"], suffixes=("_new", "_frozen"), validate="one_to_one")
    assert len(reference) == 3447
    reference["IoU_difference"] = reference.IoU_new - reference.IoU_frozen
    reference["Pointing_Game_difference"] = reference.Pointing_Game_new - reference.Pointing_Game_frozen
    discrepancies = reference[(reference.IoU_difference.abs() > 1e-12) | (reference.Pointing_Game_difference != 0)]
    discrepancies.to_csv(HERE / "localization_replay_discrepancies.csv", index=False)
    save_json(HERE / "localization_area_integrity.json", {"status": "PASS", "valid_ROI_n": 383,
        "test_images": 555, "groups": int(test.loc[valid, "group_id"].nunique()), "saved_map_hashes": source_hashes,
        "native_shape": [14,14], "evaluation_shape": [224,224], "interpolation": "bilinear_align_corners_false",
        "normalization": "per_map_minmax_threshold_only", "topk_tie_rule": "stable_row_major", "new_inference": False,
        "topk_boundary_tie_fraction": float(data.loc[data.method == "top_k_fraction", "boundary_tie"].mean()),
        "historical_localization_rows": len(original), "historical_replay_discrepancy_rows": len(discrepancies),
        "maximum_IoU_replay_difference": float(reference.IoU_difference.abs().max()), "analysis_sha256": sha(Path(__file__)),
        "output_hashes": {name: sha(HERE / name) for name in ["localization_area_per_image.csv", "localization_area_control.csv", "localization_area_control_summary.csv", "localization_paired_comparisons.csv", "localization_replay_discrepancies.csv"]}})
    print("LOCALIZATION_AREA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

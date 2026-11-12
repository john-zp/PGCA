"""Read-only historical integrity checks and strict Query isolation tests."""
import copy
import json
import subprocess

from query_model import BASE, HERE, QueryCohort, inputs, make_model, query_model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_one import Cohort, set_seed, sha, save_json


def main():
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    lock = json.loads((BASE / "source_lock.json").read_text())
    for relative, expected in lock["dataset_hashes"].items():
        assert sha(BASE / relative) == expected, relative
    for filename, expected in lock["dependency_hashes"].items():
        assert sha(__import__("pathlib").Path(filename)) == expected, filename
    for name in ["train_one.py", "experiment_models.py", "evaluation.py", "common_config.json"]:
        assert sha(BASE / name) == lock["code_hashes"][name], name
    frame = pd.read_csv(BASE / "data/dataset_manifest.csv")
    assert frame.image_id.is_unique
    assert frame.groupby("group_id").split.nunique().max() == 1
    assert frame.groupby("patient_key").split.nunique().max() == 1
    assert frame.split.value_counts().to_dict() == {"train": 2599, "val": 555, "test": 555}
    arrays = {name: np.load(BASE / "data" / (name + ".npy"), mmap_mode="r") for name in ["images", "roi_masks", "features36", "pathology_embeddings", "y"]}
    for name, value in arrays.items():
        assert len(value) == 3709 and np.isfinite(value).all(), name
    assert arrays["features36"].shape == (3709, 36)
    assert np.isin(arrays["features36"], [0, 1]).all()
    assert arrays["pathology_embeddings"].shape == (3709, 512)
    assert np.isin(arrays["y"], range(1, 7)).all()
    test_indices = frame.index[frame.split == "test"].to_numpy()
    roi_valid = arrays["roi_masks"][test_indices].reshape(555, -1).sum(1) > 0
    assert roi_valid.sum() == 383
    existing_roi = pd.read_csv(BASE / "mechanism_validation_20261014/localization_roi_subset.csv")
    assert set(existing_roi.image_id) == set(frame.iloc[test_indices[roi_valid]].image_id)
    design = HERE / "design"
    design.mkdir(exist_ok=True)
    donors = np.arange(len(frame))
    permutation_records = []
    for split, seed in zip(["train", "val", "test"], [724201, 724202, 724203]):
        indices = frame.index[frame.split == split].to_numpy()
        groups = frame.group_id.to_numpy()
        rng = np.random.default_rng(seed)
        for attempt in range(100000):
            selected = rng.permutation(indices)
            if np.all(groups[indices] != groups[selected]):
                break
        else:
            raise RuntimeError("Cannot form cross-group split-local permutation")
        donors[indices] = selected
        permutation_records.extend({"index": int(i), "donor_index": int(j), "image_id": frame.iloc[i].image_id,
            "donor_image_id": frame.iloc[j].image_id, "group_id": groups[i], "donor_group_id": groups[j],
            "split": split, "permutation_seed": seed, "rejection_attempts": attempt + 1} for i, j in zip(indices, selected))
    np.save(design / "query_donors.npy", donors, allow_pickle=False)
    pd.DataFrame(permutation_records).to_csv(design / "query_permutations.csv", index=False)
    mean = arrays["pathology_embeddings"][frame.split == "train"].astype(np.float64).mean(0).astype(np.float32)
    np.save(design / "train_mean_embedding.npy", mean, allow_pickle=False)
    frame.iloc[test_indices[roi_valid]][["image_id", "group_id"]].to_csv(design / "roi_subset.csv", index=False)
    (HERE / "split_manifest.csv").write_bytes((BASE / "split_manifest.csv").read_bytes())
    batch = next(iter(DataLoader(Cohort("train", 42), batch_size=4)))
    x = inputs(batch)
    set_seed(42)
    old = make_model("PGCA-0").cuda().eval()
    set_seed(42)
    model = query_model().cuda().eval()
    assert list(old.state_dict()) == list(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.state_dict().items())
    original, matched = old(**x), model(**x)
    for key in ["logits", "attn_map", "fusion_out"]:
        assert torch.equal(original[key], matched[key]), key
    original["loss_ord"].backward()
    matched["loss_ord"].backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(old.parameters(), model.parameters()) if p.grad is not None)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        explicit = model(**x, query_emb=x["phrase_emb"])
        assert torch.equal(matched["logits"], explicit["logits"])
        query = x["phrase_emb"].flip(0)
        changed = model(**x, query_emb=query)
        assert not torch.equal(changed["attn_map"], matched["attn_map"])
        tokens, _, _ = model.image_encoder(x["image"])
        ca = model.cross_attn
        r = ca.q_proj(model.prompt_proj(x["phrase_emb"])).unsqueeze(1)
        q = ca.q_proj(model.prompt_proj(query)).unsqueeze(1)
        v, _ = ca.mha(q, ca.k_proj(tokens), ca.v_proj(tokens), need_weights=True, average_attn_weights=False)
        expected = ca.ln1(r + v)
        expected = ca.ln2(expected + ca.ffn(expected)).squeeze(1)
        feature = model.feature_encoder(x["features"])
        global_image = model.image_encoder(x["image"])[1]
        expected_logits = model.head(model.fusion(torch.cat([global_image, expected, feature], 1)))["logits"]
        assert torch.equal(expected_logits, changed["logits"])
    smoke = {}
    for condition in ["Shuffled", "Constant"]:
        set_seed(42)
        check = query_model().cuda().train()
        optim = torch.optim.AdamW(check.parameters(), lr=.0003, weight_decay=.0001)
        losses = []
        for i, data in enumerate(DataLoader(QueryCohort("train", 42, condition), batch_size=4)):
            optim.zero_grad(set_to_none=True)
            out = check(**inputs(data))
            assert out["logits"].shape == (4, 5) and out["grid_hw"] == (14, 14)
            assert torch.isfinite(out["loss_ord"])
            out["loss_ord"].backward()
            assert all(torch.isfinite(p.grad).all() for p in check.parameters() if p.grad is not None)
            for prefix in ["image_encoder", "prompt_proj", "feature_encoder", "cross_attn.q_proj", "cross_attn.k_proj"]:
                assert any(p.grad is not None and p.grad.abs().max() > 0 for n, p in check.named_parameters() if n.startswith(prefix))
            optim.step()
            losses.append(float(out["loss_ord"].detach()))
            if i == 1:
                break
        smoke[condition] = losses
    historical = {}
    for seed in [42, 3407, 2026]:
        run = BASE / "runs/PGCA-0" / f"seed_{seed}"
        assert (run / "completed.json").is_file()
        config = json.loads((run / "config.json").read_text())
        assert config["dataset_hashes"] == lock["dataset_hashes"]
        assert config["model"] == "PGCA-0" and config["seed"] == seed
        assert not config["permuted_training_pathology"] and config["epochs"] == 30
        historical[str(seed)] = {name: sha(run / name) for name in ["best_checkpoint.pt", "test_predictions.csv", "test_attention_maps.npy", "config.json"]}
    freeze = subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)
    (HERE / "environment_semantic.txt").write_text(freeze)
    report = {"status": "PASS", "split_counts": frame.split.value_counts().to_dict(), "valid_test_roi": 383,
        "identical_matched_state_forward_backward": True, "matched_residual_manual_equivalence": True,
        "query_changes_attention": True, "two_batch_losses": smoke,
        "parameter_count": sum(p.numel() for p in model.parameters()), "matched_reused_hashes": historical,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "base_source_lock": lock,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "code_hashes": {name: sha(HERE / name) for name in ["query_model.py", "preflight_semantic.py", "run_semantic.py", "protocol.md"]},
        "design_hashes": {p.name: sha(p) for p in design.iterdir() if p.is_file()}}
    save_json(HERE / "preflight_semantic.json", report)
    print(json.dumps({k:v for k,v in report.items() if k in ["status", "split_counts", "valid_test_roi", "parameter_count", "two_batch_losses"]}), flush=True)


if __name__ == "__main__":
    main()

"""Bounded validation-only tuning, then three selected-seed evaluations."""
import argparse
import json
import subprocess
import sys
import time

from query_model import BASE, HERE
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from strong_models import PretrainedFusion, WEIGHTS
from train_one import Cohort, inputs, evaluate, set_seed, save_json, save_checkpoint, sha
from evaluation import metrics


def finite_backward(loss, model):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("Nonfinite gradient")


def train_neural(name, seed, lr):
    output = HERE / "strong_runs" / name / f"lr_{lr:g}" / f"seed_{seed}"
    if (output / "trained.json").exists():
        return output
    if output.exists():
        raise RuntimeError("Incomplete run preserved: " + str(output))
    output.mkdir(parents=True)
    config = json.loads((BASE / "common_config.json").read_text())
    config.update(model=name, seed=seed, learning_rate=lr, pretrained_weights=str(WEIGHTS[name]),
        full_backbone_finetuning=True, input_preprocessing="cached_224_grayscale_repeat_RGB_ImageNet_normalize_no_extra_crop",
        effective_lambda_static_OT=0, effective_lambda_SB_path=0, exploratory=True,
        source_lock=json.loads((HERE / "strong_source_lock.json").read_text()),
        pretrained_provenance=json.loads((HERE / "pretrained_provenance.json").read_text()))
    save_json(output / "config.json", config)
    (output / "split_manifest.csv").write_bytes((BASE / "split_manifest.csv").read_bytes())
    train, val = [Cohort(s, seed) for s in ["train", "val"]]
    set_seed(seed)
    model = PretrainedFusion(name).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    smoke = []
    for i, data in enumerate(DataLoader(train, batch_size=4)):
        optimizer.zero_grad(set_to_none=True)
        out = model(**inputs(data))
        assert out["logits"].shape == (4, 5)
        finite_backward(out["loss_ord"], model)
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.backbone.parameters())
        optimizer.step()
        smoke.append(float(out["loss_ord"].detach()))
        if i == 1:
            break
    save_json(output / "gpu_smoke.json", {"status": "PASS", "two_losses": smoke})
    del model, optimizer
    torch.cuda.empty_cache()
    set_seed(seed)
    model = PretrainedFusion(name).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    best, best_epoch, curve = (-float("inf"), -float("inf")), 0, []
    for epoch in range(1, 31):
        set_seed(seed + epoch)
        train.epoch = epoch
        model.train()
        started, total, count = time.monotonic(), 0., 0
        loader = DataLoader(train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
        for data in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs(data))
            finite_backward(out["loss_ord"], model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(out["loss_ord"].detach()) * len(data["y"])
            count += len(data["y"])
        validation = evaluate(model, val, None, "val")
        key = (validation["QWK"], -validation["MAE"])
        if key > best:
            best, best_epoch = key, epoch
            save_checkpoint(output / "best_checkpoint.pt", {"model": model.state_dict(), "epoch": epoch, "config": config, "validation": validation})
            save_json(output / "best_validation_metrics.json", validation)
        curve.append({"epoch": epoch, "training_loss": total/count, "validation": validation, "elapsed_seconds": time.monotonic()-started})
        save_json(output / "training_curve.json", curve)
        print(json.dumps({"model": name, "seed": seed, "lr": lr, "epoch": epoch, "val_QWK": validation["QWK"], "seconds": curve[-1]["elapsed_seconds"]}), flush=True)
    save_json(output / "trained.json", {"best_epoch": best_epoch, "validation_key": best, "training_seconds": sum(r["elapsed_seconds"] for r in curve), "seed": seed, "lr": lr})
    return output


def neural_workflow(name):
    candidates = [train_neural(name, 42, lr) for lr in [.0001, .0003]]
    selected = max(candidates, key=lambda p: (*json.loads((p / "trained.json").read_text())["validation_key"], -json.loads((p / "trained.json").read_text())["lr"]))
    chosen = json.loads((selected / "trained.json").read_text())
    save_json(HERE / "strong_runs" / name / "selection.json", {"selected_lr": chosen["lr"], "selection_split": "val", "candidates": [str(p.relative_to(HERE)) for p in candidates], "test_seen": False})
    outputs = [selected] + [train_neural(name, seed, chosen["lr"]) for seed in [3407, 2026]]
    for output in outputs:
        if (output / "completed.json").exists():
            continue
        checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model = PretrainedFusion(name).cuda()
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, Cohort("test", checkpoint["config"]["seed"]), output, "test")
        save_json(output / "completed.json", {"model": name, "seed": checkpoint["config"]["seed"], "best_epoch": checkpoint["epoch"], "test": result,
            "checkpoint_sha256": sha(output / "best_checkpoint.pt"), "exploratory": True})
        del model
        torch.cuda.empty_cache()


def train_catboost(seed, depth, lr):
    sys.path.insert(0, str(HERE / "supplement_packages"))
    from catboost import CatBoostClassifier
    directory = HERE / "strong_runs/CatBoost-F" / f"depth_{depth}_lr_{lr:g}" / f"seed_{seed}"
    if (directory / "trained.json").exists():
        return directory
    if directory.exists():
        raise RuntimeError("Incomplete CatBoost run preserved: " + str(directory))
    directory.mkdir(parents=True)
    frame = pd.read_csv(BASE / "data/dataset_manifest.csv")
    x, y = [np.load(BASE / "data" / (n + ".npy")) for n in ["features36", "y"]]
    train, val = frame.split == "train", frame.split == "val"
    config = dict(iterations=1000, depth=depth, learning_rate=lr, l2_leaf_reg=3,
        random_seed=seed, thread_count=8, loss_function="MultiClass", boosting_type="Plain",
        task_type="CPU", verbose=False, allow_writing_files=False)
    save_json(directory / "config.json", {**config, "exploratory": True, "source_lock": json.loads((HERE / "strong_source_lock.json").read_text())})
    (directory / "split_manifest.csv").write_bytes((BASE / "split_manifest.csv").read_bytes())
    started = time.monotonic()
    model = CatBoostClassifier(**config)
    model.fit(x[train], y[train])
    assert set(model.classes_) == set(range(1, 7))
    best, epoch, curve = (-float("inf"), -float("inf")), 0, []
    for step, predicted in enumerate(model.staged_predict(x[val], prediction_type="Class", eval_period=1), 1):
        value = metrics(y[val], np.asarray(predicted).reshape(-1).astype(int))
        key = (value["QWK"], -value["MAE"])
        curve.append({"iteration": step, "validation": value})
        if key > best:
            best, epoch = key, step
    model.save_model(str(directory / "all_trees.cbm"))
    model.shrink(epoch)
    model.save_model(str(directory / "best_checkpoint.cbm"))
    save_json(directory / "training_curve.json", curve)
    save_json(directory / "best_validation_metrics.json", curve[epoch-1]["validation"])
    save_json(directory / "trained.json", {"best_epoch": epoch, "validation_key": best, "training_seconds": time.monotonic()-started, "seed": seed, "depth": depth, "lr": lr})
    print(json.dumps({"model": "CatBoost-F", "seed": seed, "depth": depth, "lr": lr, "best_epoch": epoch, "validation_key": best}), flush=True)
    return directory


def catboost_workflow():
    candidates = [train_catboost(42, depth, lr) for depth in [4, 6, 8] for lr in [.03, .1]]
    selected = max(candidates, key=lambda p: tuple(json.loads((p / "trained.json").read_text())["validation_key"]))
    best = json.loads((selected / "trained.json").read_text())
    save_json(HERE / "strong_runs/CatBoost-F/selection.json", {"selected_depth": best["depth"], "selected_lr": best["lr"], "selection_split": "val", "test_seen": False, "candidates": [str(p.relative_to(HERE)) for p in candidates]})
    outputs = [selected] + [train_catboost(seed, best["depth"], best["lr"]) for seed in [3407, 2026]]
    from catboost import CatBoostClassifier
    frame = pd.read_csv(BASE / "data/dataset_manifest.csv")
    subset = frame[frame.split == "test"]
    x = np.load(BASE / "data/features36.npy")[subset.index]
    y = np.load(BASE / "data/y.npy")[subset.index]
    for output in outputs:
        model = CatBoostClassifier().load_model(str(output / "best_checkpoint.cbm"))
        predicted = np.asarray(model.predict(x, prediction_type="Class")).reshape(-1).astype(int)
        result = metrics(y, predicted)
        rows = subset[["image_id", "patient_key", "group_id"]].copy()
        rows["y_true"], rows["y_pred"] = y, predicted
        rows.to_csv(output / "test_predictions.csv", index=False)
        np.savetxt(output / "test_confusion_matrix.csv", result["confusion_matrix"], delimiter=",", fmt="%d")
        save_json(output / "test_metrics.json", result)
        trained = json.loads((output / "trained.json").read_text())
        save_json(output / "completed.json", {"model": "CatBoost-F", "seed": trained["seed"], "best_epoch": trained["best_epoch"], "test": result,
            "checkpoint_sha256": sha(output / "best_checkpoint.cbm"), "exploratory": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(WEIGHTS) + ["CatBoost-F"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    lock = json.loads((HERE / "strong_source_lock.json").read_text())
    for name, value in lock["code_hashes"].items():
        assert sha(HERE / name) == value, name
    for name, value in lock["dataset_hashes"].items():
        assert sha(BASE / name) == value, name
    provenance = json.loads((HERE / "pretrained_provenance.json").read_text())
    assert provenance["status"] == "PASS"
    assert sha(HERE / "pretrained_provenance.json") == lock["pretrained_provenance_sha256"]
    for record in provenance["weights"].values():
        assert sha(HERE / "pretrained_cache/checkpoints" / record["url"].rsplit("/", 1)[-1]) == record["sha256"]
    if args.model == "CatBoost-F":
        catboost_workflow()
    else:
        neural_workflow(args.model)

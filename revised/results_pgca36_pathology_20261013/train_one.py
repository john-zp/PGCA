"""One versioned run on the locked cohort; evaluation never selects settings."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from evaluation import metrics, threshold_calibration
from experiment_models import make_model
from spatial_losses import alignment


OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Cohort(Dataset):
    def __init__(self, split, seed, shuffled=False):
        self.manifest = pd.read_csv(OUT / "data/dataset_manifest.csv")
        self.indices = self.manifest.index[self.manifest["split"] == split].to_numpy()
        self.rows = self.manifest.to_dict("records")
        self.arrays = {name: np.load(OUT / "data" / (name + ".npy"), mmap_mode="r") for name in ["images", "roi_masks", "features36", "pathology_embeddings", "y"]}
        self.split, self.seed, self.epoch = split, seed, 0
        self.donors = {int(index): int(index) for index in self.indices}
        if shuffled and split == "train":
            order = np.random.default_rng(42).permutation(self.indices)
            self.donors = {int(index): int(donor) for index, donor in zip(order, np.roll(order, 1))}
        self.shuffled = shuffled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        image = np.array(self.arrays["images"][index], dtype=np.float32, copy=True)
        roi = np.array(self.arrays["roi_masks"][index], dtype=np.float32, copy=True)
        if self.split == "train":
            token = f"{self.seed}|{self.epoch}|{self.rows[index]['image_id']}".encode()
            if hashlib.sha256(token).digest()[0] < 128:
                image, roi = np.flip(image, axis=-1).copy(), np.flip(roi, axis=-1).copy()
        return {"image": torch.from_numpy(image), "roi": torch.from_numpy(roi),
                "features": torch.from_numpy(np.array(self.arrays["features36"][index], copy=True)),
                "phrase_emb": torch.from_numpy(np.array(self.arrays["pathology_embeddings"][self.donors[index]], copy=True)),
                "y": torch.tensor(int(self.arrays["y"][index]), dtype=torch.long), "index": index}


def inputs(data):
    return {key: data[key].to("cuda") for key in ["image", "features", "phrase_emb", "y"]}


def loss_for(out, roi, name, config):
    loss = out["loss_ord"]
    details = {"valid_roi": 0, "marginal_error": 0.0, "endpoint_error": 0.0}
    if name in ["PGCA-staticOT", "PGCA-full-SBpath"]:
        static, path, details = alignment(out["attn_map"], roi.to("cuda"), out["grid_hw"],
            name == "PGCA-full-SBpath", config["OT_epsilon"], config["sinkhorn_iterations"], config["static_OT_internal_beta"])
        loss = loss + config["lambda_static_OT"] * static
        if name == "PGCA-full-SBpath":
            loss = loss + config["lambda_SB_path"] * path
    return loss, details


def gpu_smoke(name, train, config, seed):
    set_seed(seed)
    model = make_model(name).to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    start = time.monotonic()
    records = []
    for number, data in enumerate(DataLoader(train, batch_size=4, shuffle=False)):
        optimizer.zero_grad(set_to_none=True)
        out = model(**inputs(data))
        loss, details = loss_for(out, data["roi"], name, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite CUDA smoke loss")
        loss.backward()
        if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
            raise FloatingPointError("Nonfinite CUDA smoke gradient")
        optimizer.step()
        records.append({"batch": number, "loss": float(loss.detach()), **details})
        if number == 1:
            break
    torch.cuda.synchronize()
    return {"status": "PASS", "batches": records, "elapsed_seconds": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def evaluate(model, dataset, run, prefix, with_maps=False):
    model.eval()
    predictions, truth, rows, probabilities, maps, localization = [], [], [], [], [], []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=16, shuffle=False):
            predictor_inputs = inputs(data)
            predictor_inputs.pop("y")
            out = model(**predictor_inputs)
            pred, actual = out["pred"].cpu().numpy(), data["y"].numpy()
            p_ge = out["p_ge"].cpu().numpy()
            predictions.extend(pred.tolist())
            truth.extend(actual.tolist())
            probabilities.extend(p_ge.tolist())
            for position, index in enumerate(data["index"].tolist()):
                record = dataset.rows[index]
                rows.append({"image_id": record["image_id"], "patient_key": record["patient_key"], "group_id": record["group_id"],
                    "y_true": int(actual[position]), "y_pred": int(pred[position]),
                    **{f"p_ge_{threshold}": float(p_ge[position, threshold - 2]) for threshold in range(2, 7)}})
            if with_maps and "attn_map" in out:
                h, w = out["grid_hw"]
                attention = out["attn_map"].view(-1, 1, h, w).cpu()
                maps.extend(attention[:, 0].numpy())
                large = F.interpolate(attention, size=(224, 224), mode="bilinear", align_corners=False).numpy()[:, 0]
                for position, index in enumerate(data["index"].tolist()):
                    roi = data["roi"][position, 0].numpy() > 0
                    if not roi.any():
                        continue
                    heatmap = large[position]
                    normalized = (heatmap - heatmap.min()) / max(float(heatmap.max() - heatmap.min()), 1e-12)
                    selected = normalized >= .5
                    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    localization.append({"image_id": dataset.rows[index]["image_id"], "group_id": dataset.rows[index]["group_id"],
                        "pointing_game_hit": int(roi[peak]), "IoU_threshold_0_5": float(np.logical_and(roi, selected).sum() / np.logical_or(roi, selected).sum()),
                        "classification_correct": int(actual[position] == pred[position])})
    summary = metrics(truth, predictions)
    summary["calibration"] = threshold_calibration(truth, probabilities)
    if run is not None:
        write_csv(run / (prefix + "_predictions.csv"), rows)
        np.savetxt(run / (prefix + "_confusion_matrix.csv"), np.asarray(summary["confusion_matrix"]), delimiter=",", fmt="%d")
        save_json(run / (prefix + "_metrics.json"), summary)
        if maps:
            np.save(run / (prefix + "_attention_maps.npy"), np.stack(maps), allow_pickle=False)
        if localization:
            write_csv(run / (prefix + "_localization.csv"), localization)
            save_json(run / (prefix + "_localization_summary.json"), {"valid_ROI_n": len(localization),
                "pointing_game": float(np.mean([row["pointing_game_hit"] for row in localization])),
                "IoU_threshold_0_5": float(np.mean([row["IoU_threshold_0_5"] for row in localization])),
                "interpretation": "spatial correspondence, not causal explanation fidelity"})
            failures = [row for row in localization if not row["pointing_game_hit"] or not row["classification_correct"]]
            if failures:
                write_csv(run / (prefix + "_localization_failures.csv"), failures)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_ALLOCATED: formal training has no CPU fallback")
    config = json.loads((OUT / "common_config.json").read_text())
    if args.seed not in config["seeds"]:
        raise ValueError("Seed is not part of the frozen experiment protocol")
    source = json.loads((OUT / "source_lock.json").read_text())
    gate = json.loads((OUT / "preflight_checks.json").read_text())
    if gate["models"][args.model]["status"] != "PASS_CPU_TWO_BATCHES":
        raise RuntimeError("DEPENDENT_PREFLIGHT_BLOCKED")
    if args.model.startswith("PGCA-") and gate["overfit50"]["status"] != "PASS":
        raise RuntimeError("OVERFIT50_GATE_BLOCKED")
    for relative, expected in source["code_hashes"].items():
        if sha(OUT / relative) != expected:
            raise RuntimeError("Source code changed after lock: " + relative)
    for relative, expected in source["dataset_hashes"].items():
        if sha(OUT / relative) != expected:
            raise RuntimeError("Locked dataset changed: " + relative)
    for filename, expected in source["dependency_hashes"].items():
        if sha(Path(filename)) != expected:
            raise RuntimeError("Frozen graph dependency changed: " + filename)
    freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout.strip() + "\n"
    if hashlib.sha256(freeze.encode()).hexdigest() != source["environment_sha256"]:
        raise RuntimeError("Runtime dependencies changed after environment freeze")
    name = args.model.replace("+", "_") + ("_train_permuted_pathology" if args.permuted else "")
    run = OUT / "runs" / name / ("seed_" + str(args.seed))
    run.mkdir(parents=True, exist_ok=True)
    if (run / "completed.json").is_file():
        print("ALREADY_COMPLETE", flush=True)
        return
    config.update({"model": args.model, "seed": args.seed, "permuted_training_pathology": args.permuted,
                   "code_commit": source["code_commit"], "frozen_dependency_commit": source["frozen_dependency_commit"],
                   "dataset_hashes": source["dataset_hashes"]})
    save_json(run / "config.json", config)
    save_json(run / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "source_lock": source})
    (run / "split_manifest.csv").write_bytes((OUT / "split_manifest.csv").read_bytes())
    train, val, test = Cohort("train", args.seed, args.permuted), Cohort("val", args.seed), Cohort("test", args.seed)
    if args.permuted:
        write_csv(run / "training_pathology_permutation.csv", [{"image_id": train.rows[index]["image_id"], "donor_image_id": train.rows[donor]["image_id"], "split": "train"} for index, donor in train.donors.items()])
    save_json(run / "gpu_smoke.json", gpu_smoke(args.model, train, config, args.seed))
    set_seed(args.seed)
    model = make_model(args.model).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    first, best, best_epoch = 1, (-float("inf"), -float("inf")), 0
    curve = []
    if (run / "last_checkpoint.pt").is_file():
        resume = torch.load(run / "last_checkpoint.pt", map_location="cuda", weights_only=False)
        if resume["config"]["code_commit"] != config["code_commit"] or resume["config"]["dataset_hashes"] != config["dataset_hashes"]:
            raise RuntimeError("Resume checkpoint does not match locked code and cohort")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        first, best, best_epoch = resume["epoch"] + 1, tuple(resume["selection"]), resume["best_epoch"]
        curve = json.loads((run / "training_curve.json").read_text())
    for epoch in range(first, config["epochs"] + 1):
        set_seed(args.seed + epoch)
        train.epoch = epoch
        model.train()
        total, examples, max_marginal, max_endpoint, valid_roi = 0., 0, 0., 0., 0
        started = time.monotonic()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        for data in DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs(data))
            loss, details = loss_for(out, data["roi"], args.model, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise FloatingPointError("Nonfinite training gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total += float(loss.detach()) * len(data["y"])
            examples += len(data["y"])
            max_marginal = max(max_marginal, details["marginal_error"])
            max_endpoint = max(max_endpoint, details["endpoint_error"])
            valid_roi += details["valid_roi"]
        validation = evaluate(model, val, None, "val")
        selection = (validation["QWK"], -validation["MAE"])
        if selection > best:
            best, best_epoch = selection, epoch
            save_checkpoint(run / "best_checkpoint.pt", {"model": model.state_dict(), "epoch": epoch, "config": config, "validation": validation})
            save_json(run / "best_validation_metrics.json", validation)
        curve.append({"epoch": epoch, "training_loss": total / examples, "validation": validation, "elapsed_seconds": time.monotonic() - started,
                      "valid_training_ROI": valid_roi, "maximum_marginal_error": max_marginal, "maximum_endpoint_error": max_endpoint})
        save_json(run / "training_curve.json", curve)
        save_checkpoint(run / "last_checkpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "selection": best, "best_epoch": best_epoch,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "config": config})
        print(json.dumps({"model": args.model, "seed": args.seed, "epoch": epoch, "training_loss": total / examples, "val_QWK": validation["QWK"], "seconds": curve[-1]["elapsed_seconds"]}), flush=True)
    checkpoint = torch.load(run / "best_checkpoint.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final = evaluate(model, test, run, "test", with_maps=True)
    save_json(run / "completed.json", {"model": args.model, "seed": args.seed, "best_epoch": best_epoch, "code_commit": source["code_commit"], "test": final,
                                      "text_role": "pathology known at training and inference", "permuted_training_pathology": args.permuted})
    print("COMPLETED " + str(run), flush=True)


if __name__ == "__main__":
    main()

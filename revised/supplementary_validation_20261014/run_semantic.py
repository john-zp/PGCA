"""Only six new Query-only ablation runs; matched historical runs stay intact."""
import argparse
import json
import time

from query_model import BASE, HERE, QueryCohort, inputs, query_model
import torch
from torch.utils.data import DataLoader
import train_one
from train_one import evaluate, save_json, save_checkpoint, set_seed, sha


def run(condition, seed):
    torch.set_num_threads(8)
    gate = json.loads((HERE / "preflight_semantic.json").read_text())
    assert gate["status"] == "PASS"
    for name, expected in gate["code_hashes"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in gate["design_hashes"].items():
        assert sha(HERE / "design" / name) == expected, name
    for name, expected in gate["base_source_lock"]["dataset_hashes"].items():
        assert sha(BASE / name) == expected, name
    train_one.inputs = inputs
    config = json.loads((BASE / "common_config.json").read_text())
    config.update(condition=condition, seed=seed, code_commit=gate["code_commit"],
        dataset_hashes=gate["base_source_lock"]["dataset_hashes"], query_design_hashes=gate["design_hashes"],
        effective_lambda_static_OT=0, effective_lambda_SB_path=0, exploration_only=True,
        direct_text_residual="matched_case_q_proj_prompt", selection_query=condition)
    output = HERE / "semantic_runs" / condition / f"seed_{seed}"
    if (output / "completed.json").exists():
        print("ALREADY_COMPLETE", flush=True)
        return
    if output.exists():
        raise RuntimeError("Refuse to overwrite incomplete run: " + str(output))
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    (output / "split_manifest.csv").write_bytes((BASE / "split_manifest.csv").read_bytes())
    train, val, test = [QueryCohort(s, seed, condition) for s in ["train", "val", "test"]]
    set_seed(seed)
    model = query_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, best_epoch, curve = (-float("inf"), -float("inf")), 0, []
    for epoch in range(1, config["epochs"] + 1):
        set_seed(seed + epoch)
        train.epoch = epoch
        model.train()
        start, loss_sum, count = time.monotonic(), 0., 0
        generator = torch.Generator().manual_seed(seed + epoch)
        for data in DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs(data))
            loss = out["loss_ord"]
            assert torch.isfinite(loss), "Nonfinite loss"
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "Nonfinite gradient"
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(data["y"])
            count += len(data["y"])
        validation = evaluate(model, val, None, "val")
        selection = (validation["QWK"], -validation["MAE"])
        if selection > best:
            best, best_epoch = selection, epoch
            save_checkpoint(output / "best_checkpoint.pt", {"model": model.state_dict(), "epoch": epoch, "config": config, "validation": validation})
            save_json(output / "best_validation_metrics.json", validation)
        curve.append({"epoch": epoch, "training_loss": loss_sum/count, "validation": validation, "elapsed_seconds": time.monotonic()-start})
        save_json(output / "training_curve.json", curve)
        save_checkpoint(output / "last_checkpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": config})
        print(json.dumps({"condition": condition, "seed": seed, "epoch": epoch, "val_QWK": validation["QWK"], "seconds": curve[-1]["elapsed_seconds"]}), flush=True)
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final = evaluate(model, test, output, "test", with_maps=True)
    save_json(output / "completed.json", {"condition": condition, "seed": seed, "best_epoch": best_epoch, "test": final,
        "training_seconds": sum(r["elapsed_seconds"] for r in curve), "checkpoint_sha256": sha(output / "best_checkpoint.pt"), "exploratory": True})
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["Shuffled", "Constant"], required=True)
    parser.add_argument("--seed", choices=[42, 3407, 2026], type=int, required=True)
    args = parser.parse_args()
    run(args.condition, args.seed)

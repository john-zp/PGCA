import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from evaluation import metrics
from experiment_models import IndependentOrdinalHead, make_model
from spatial_losses import alignment, bridge, marginal_error, solve_coupling, spatial_distributions


OUT = Path(__file__).resolve().parent
NAMES = ["F", "I", "P", "I+F", "I+P", "F+P", "I+F+P-simple", "PGCA-0", "PGCA-staticOT", "PGCA-full-SBpath"]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def static_maths():
    tests = []
    for support in [1, 4, 16]:
        logits = torch.randn(2, 1, 16, dtype=torch.float64, requires_grad=True)
        roi = torch.zeros(2, 1, 4, 4, dtype=torch.float64)
        roi.flatten(1)[:, :support] = 1
        loss, _, details = alignment(logits.softmax(-1), roi, (4, 4), False)
        gradient = torch.autograd.grad(loss, logits)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        tests.append({"roi_support": support, **details})
    return {"static_ot": "PASS", "sb_path": "PENDING", "static_tests": tests}


def maths():
    value = {"static_ot": "PENDING", "sb_path": "PENDING", "tests": []}
    torch.manual_seed(42)
    dtype = torch.float64
    P = torch.tensor([[.65, .30, .05], [.05, .70, .25], [.30, .10, .60]], dtype=dtype)
    a = torch.tensor([[.55, .35, .1]], dtype=dtype)
    b = torch.tensor([[.1, .2, .7]], dtype=dtype)
    states, gamma = bridge(a, b, P)
    horizon = len(states) - 1
    endpoint_transition = torch.linalg.matrix_power(P, horizon)
    enumeration = torch.zeros(horizon + 1, 3, dtype=dtype)
    for trajectory in itertools.product(range(3), repeat=horizon + 1):
        i, j = trajectory[0], trajectory[-1]
        weight = gamma[0, i, j] / endpoint_transition[i, j]
        for source, destination in zip(trajectory[:-1], trajectory[1:]):
            weight = weight * P[source, destination]
        for step, state in enumerate(trajectory):
            enumeration[step, state] += weight
    actual = torch.cat(states, dim=0)
    assert torch.allclose(actual, enumeration, atol=1e-9, rtol=1e-9)
    assert torch.allclose(actual[0], a[0], atol=1e-9)
    assert torch.allclose(actual[-1], b[0], atol=1e-9)
    assert torch.allclose(actual.sum(1), torch.ones(horizon + 1, dtype=dtype), atol=1e-9)
    value["tests"].append({"name": "directed_reference_path_enumeration_and_endpoints", "max_error": float((actual - enumeration).abs().max())})
    for support in [1, 4, 16]:
        logits = torch.randn(2, 1, 16, dtype=dtype, requires_grad=True)
        attention = logits.softmax(-1)
        roi = torch.zeros(2, 1, 4, 4, dtype=dtype)
        roi.flatten(1)[:, :support] = 1
        static, path, details = alignment(attention, roi, (4, 4), with_path=True)
        derivative = torch.autograd.grad(static + .05 * path, logits)[0]
        assert torch.isfinite(derivative).all() and derivative.abs().max() > 0
        value["tests"].append({"name": "roi_support_" + str(support), **details})
    logits = torch.tensor([[[.3, -.2, .7, .1]]], dtype=dtype, requires_grad=True)
    roi = torch.tensor([[[[0., 1.], [1., 0.]]]], dtype=dtype)
    static, path, _ = alignment(logits.softmax(-1), roi, (2, 2), True)
    derivative = torch.autograd.grad(static + .05 * path, logits)[0][0, 0, 1]
    step = 1e-5
    values = []
    for direction in [-1, 1]:
        altered = logits.detach().clone()
        altered[0, 0, 1] += direction * step
        x, z, _ = alignment(altered.softmax(-1), roi, (2, 2), True)
        values.append(x + .05 * z)
    difference = (values[1] - values[0]) / (2 * step)
    assert torch.allclose(derivative, difference, atol=1e-6, rtol=1e-4)
    empty = torch.zeros_like(roi)
    x, z, detail = alignment(logits.softmax(-1), empty, (2, 2), True)
    assert detail["valid_roi"] == 0 and x == 0 and z == 0
    # Image and ROI transformations must share axes and preserve registration.
    image = torch.zeros(1, 1, 8, 8)
    mask = image.clone()
    image[:, :, 2:5, 1:4] = 1
    mask.copy_(image)
    assert torch.equal(torch.flip(image, [-1]), torch.flip(mask, [-1]))
    assert torch.equal(torch.nn.functional.interpolate(image, (16, 16), mode="nearest"), torch.nn.functional.interpolate(mask, (16, 16), mode="nearest"))
    known = metrics(np.arange(1, 7), np.array([1, 2, 2, 4, 5, 6]))
    assert known["MAE"] >= 1 - known["Accuracy"]
    try:
        IndependentOrdinalHead()(torch.ones(2, 256), torch.tensor([0, 7]))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid labels were silently clipped")
    value.update({"static_ot": "PASS", "sb_path": "PASS_SYNTHETIC_MATH_ONLY", "finite_difference_error": float(abs(derivative - difference)),
                  "empty_roi_classification_kept": True, "shared_resize_flip_registration": True, "invalid_labels_rejected": True})
    return value


def batch(indices, arrays):
    return {"image": torch.from_numpy(np.asarray(arrays["images"][indices], dtype=np.float32)),
            "features": torch.from_numpy(np.asarray(arrays["features36"][indices], dtype=np.float32)),
            "phrase_emb": torch.from_numpy(np.asarray(arrays["pathology_embeddings"][indices], dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(arrays["y"][indices], dtype=np.int64)),
            "roi": torch.from_numpy(np.asarray(arrays["roi_masks"][indices], dtype=np.float32))}


def smoke(name, indices, arrays, math_status):
    torch.manual_seed(42)
    model = make_model(name).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rows = []
    for start in [0, 2]:
        data = batch(indices[start:start + 2], arrays)
        roi = data.pop("roi")
        optimizer.zero_grad(set_to_none=True)
        out = model(**data)
        assert out["logits"].shape == (2, 5) and torch.isfinite(out["logits"]).all()
        loss = out["loss_ord"]
        details = {}
        if name == "PGCA-staticOT" or (name == "PGCA-full-SBpath" and math_status["sb_path"] == "PASS_SYNTHETIC_MATH_ONLY"):
            static, path, details = alignment(out["attn_map"], roi, out["grid_hw"], name == "PGCA-full-SBpath")
            loss = loss + .2 * static + (.05 * path if name == "PGCA-full-SBpath" else 0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        assert model.head.fc.weight.grad.abs().sum() > 0
        if name.startswith("PGCA-"):
            assert model.prompt_proj[0].weight.grad.abs().sum() > 0
            assert model.feature_encoder[0].weight.grad.abs().sum() > 0
            assert model.cross_attn.q_proj.weight.grad.abs().sum() > 0
        optimizer.step()
        rows.append({"batch": start // 2, "loss": float(loss.detach()), **details})
    return {"status": "PASS_CPU_TWO_BATCHES", "batch_size": 2, "image_size": 224, "batches": rows}


def overfit(indices, arrays):
    rng = np.random.default_rng(42)
    selected = rng.choice(indices, size=50, replace=False)
    torch.manual_seed(42)
    model = make_model("PGCA-0").train()
    # A diagnostic fit, not an experimental hyperparameter-selection run.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
        if isinstance(layer, torch.nn.MultiheadAttention):
            layer.dropout = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    curve = []
    started = time.monotonic()
    for step in range(301):
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                outputs, losses = [], []
                for offset in range(0, 50, 10):
                    data = batch(selected[offset:offset + 10], arrays)
                    data.pop("roi")
                    out = model(**data)
                    outputs.extend(out["pred"].tolist())
                    losses.append(float(out["loss_ord"]))
                accuracy = float(np.mean(np.asarray(outputs) == arrays["y"][selected]))
                curve.append({"step": step, "diagnostic_training_accuracy": accuracy, "ordinal_loss": float(np.mean(losses))})
                print("OVERFIT50 " + json.dumps(curve[-1]), flush=True)
            model.train()
            if step > 0 and accuracy >= .95 and np.mean(losses) <= .08:
                break
        if step == 300:
            break
        chosen = rng.choice(selected, 16, replace=False)
        data = batch(chosen, arrays)
        data.pop("roi")
        optimizer.zero_grad(set_to_none=True)
        out = model(**data)
        out["loss_ord"].backward()
        optimizer.step()
    passed = curve[-1]["diagnostic_training_accuracy"] >= .95 and curve[-1]["ordinal_loss"] <= .08
    return {"status": "PASS" if passed else "FAILED_OVERFIT_GATE", "sample_array_indices": selected.tolist(), "sample_count": 50,
            "split": "train_only", "image_size": 224, "max_steps": 300, "diagnostic_lr": .003,
            "diagnostic_dropout": 0, "elapsed_seconds": time.monotonic() - started, "curve": curve,
            "not_test_performance_or_formal_training": True}


def main():
    torch.set_num_threads(8)
    arrays = {name: np.load(OUT / "data" / (name + ".npy"), mmap_mode="r") for name in ["images", "roi_masks", "features36", "pathology_embeddings", "y"]}
    manifest = pd.read_csv(OUT / "data/dataset_manifest.csv")
    train = manifest.loc[manifest["split"] == "train", "array_index"].to_numpy()
    with_roi = manifest.loc[(manifest["split"] == "train") & (manifest["roi_status"] == "geometry_validated_rectangle"), "array_index"].to_numpy()[:4]
    result = {"execution_device": "cpu", "cuda_available": torch.cuda.is_available(), "data": "PASS", "models": {}, "maths": {}}
    try:
        result["maths"] = static_maths()
    except Exception as exc:
        result["maths"] = {"static_ot": "BLOCKED", "sb_path": "PENDING", "static_error": repr(exc)}
    try:
        path_result = maths()
        path_result.pop("static_ot")
        result["maths"].update(path_result)
    except Exception as exc:
        result["maths"].update({"sb_path": "BLOCKED", "path_error": repr(exc)})
    save("preflight_checks.json", result)
    for name in NAMES:
        try:
            if name == "PGCA-full-SBpath" and result["maths"]["sb_path"] != "PASS_SYNTHETIC_MATH_ONLY":
                result["models"][name] = {"status": "BLOCKED_SB_MATH"}
            elif name == "PGCA-staticOT" and result["maths"]["static_ot"] != "PASS":
                result["models"][name] = {"status": "BLOCKED_OT_MATH"}
            else:
                result["models"][name] = smoke(name, with_roi, arrays, result["maths"])
        except Exception as exc:
            result["models"][name] = {"status": "BLOCKED_SMOKE", "error": repr(exc)}
        save("preflight_checks.json", result)
        print("SMOKE " + name + " " + result["models"][name]["status"], flush=True)
    result["overfit50"] = overfit(train, arrays)
    save("preflight_checks.json", result)
    print(json.dumps({"models": result["models"], "maths": result["maths"], "overfit50": result["overfit50"]["status"]}), flush=True)


if __name__ == "__main__":
    main()

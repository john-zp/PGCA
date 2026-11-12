from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FILE_DIR))

from datasets.multimodal_us import DatasetConfig, MultimodalUSDataset
from losses.sb import sb_losses
from utils.metrics import compute_metrics, save_json


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_xy_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).view(-1, 2)


def build_loaders(cfg):
    split_dir = Path(cfg["paths"]["split_dir"])
    task = cfg["experiment"]["task"]
    image_size = cfg.get("model", {}).get("image_size", 384)

    common = {
        "task": task,
        "data_root": cfg["paths"]["data_root"],
        "labels_csv": cfg["paths"].get("labels_csv"),
        "feature_xlsx": cfg["paths"].get("feature_xlsx"),
        "phrase_xlsx": cfg["paths"].get("phrase_xlsx"),
        "image_size": image_size,
    }

    ds_train = MultimodalUSDataset(
        DatasetConfig(split_file=str(split_dir / "train_ids.txt"), **common)
    )
    ds_val = MultimodalUSDataset(
        DatasetConfig(split_file=str(split_dir / "val_ids.txt"), **common)
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    return ds_train, ds_val, dl_train, dl_val


def infer_dims(dataset: MultimodalUSDataset):
    sample = dataset[0]
    feature_dim = int(sample["features"].numel()) if "features" in sample else None
    phrase_dim = int(sample["phrase_emb"].numel()) if "phrase_emb" in sample else None
    return feature_dim, phrase_dim


def build_model(cfg, dataset):
    task = cfg["experiment"]["task"]
    feature_dim, phrase_dim = infer_dims(dataset)
    model_cfg = cfg["model"]
    num_classes = int(model_cfg["num_classes"])

    if task == "feature_only":
        from models.feature_model import FeatureOnlyModel

        return FeatureOnlyModel(
            feature_dim=feature_dim,
            hidden_dims=list(model_cfg.get("hidden_dims", [128, 64])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    if task == "image_only":
        from models.image_model import ImageOnlyModel

        return ImageOnlyModel(
            image_embed_dim=int(model_cfg.get("image_embed_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    if task == "image_feature":
        from models.image_feature_model import ImageFeatureModel

        return ImageFeatureModel(
            feature_dim=feature_dim,
            image_embed_dim=int(model_cfg.get("image_embed_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    if task == "phrase_only":
        from models.phrase_model import PhraseOnlyModel

        return PhraseOnlyModel(
            phrase_dim=phrase_dim,
            hidden_dims=list(model_cfg.get("hidden_dims", [256, 128])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    if task == "image_phrase":
        from models.image_phrase_model import ImagePhraseModel

        return ImagePhraseModel(
            phrase_dim=phrase_dim,
            image_embed_dim=int(model_cfg.get("image_embed_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            attn_heads=int(model_cfg.get("attn_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    if task == "all_input":
        from models.pgca_model import PGCAModel

        return PGCAModel(
            phrase_dim=phrase_dim,
            feature_dim=feature_dim,
            image_embed_dim=int(model_cfg.get("image_embed_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            attn_heads=int(model_cfg.get("attn_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown task: {task}")


def compute_batch_loss(cfg, model, batch, device):
    task = cfg["experiment"]["task"]
    loss_cfg = cfg["loss"]
    y = batch["y"].to(device)

    if task == "feature_only":
        out = model(batch["features"].float().to(device), y=y)
        loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]
        return out, loss

    if task == "image_only":
        out = model(batch["image"].float().to(device), y=y)
        loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]
        return out, loss

    if task == "image_feature":
        out = model(
            batch["image"].float().to(device),
            batch["features"].float().to(device),
            y=y,
        )
        loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]
        return out, loss

    if task == "phrase_only":
        out = model(batch["phrase_emb"].float().to(device), y=y)
        loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]
        return out, loss

    if task == "image_phrase":
        out = model(
            batch["image"].float().to(device),
            batch["phrase_emb"].float().to(device),
            y=y,
        )
        loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]
        return out, loss

    image = batch["image"].float().to(device)
    features = batch["features"].float().to(device)
    phrase_emb = batch["phrase_emb"].float().to(device)
    roi_mask = batch["roi_mask"].float().to(device)
    roi_valid = batch["roi_valid"].float().to(device)

    out = model(image=image, features=features, phrase_emb=phrase_emb, y=y)
    loss = float(loss_cfg.get("lambda_ord", 1.0)) * out["loss_ord"]

    if (
        float(loss_cfg.get("lambda_sb_static", 0.0)) > 0
        or float(loss_cfg.get("lambda_sb_path", 0.0)) > 0
    ) and roi_valid.sum() > 0:
        attn = out["attn_map"].squeeze(1)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-12)

        h, w = out["grid_hw"]
        roi_small = F.interpolate(roi_mask, size=(h, w), mode="nearest")
        roi_dist = roi_small.view(roi_small.size(0), -1)

        valid_mask = (roi_valid > 0.5) & (roi_dist.sum(dim=1) > 0)
        if valid_mask.any():
            roi_dist = roi_dist[valid_mask]
            roi_dist = roi_dist / (roi_dist.sum(dim=1, keepdim=True) + 1e-12)
            attn = attn[valid_mask]

            xy = build_xy_grid(h, w, device)
            sb_s, sb_p = sb_losses(
                attn,
                roi_dist,
                xy,
                eps=float(loss_cfg.get("sb_eps", 0.05)),
                iters=int(loss_cfg.get("sb_iters", 50)),
                steps=int(loss_cfg.get("sb_steps", 3)),
            )

            loss = (
                loss
                + float(loss_cfg.get("lambda_sb_static", 0.0)) * sb_s
                + float(loss_cfg.get("lambda_sb_path", 0.0)) * sb_p
            )
            out["loss_sb_static"] = sb_s.detach()
            out["loss_sb_path"] = sb_p.detach()

    return out, loss


def evaluate(cfg, model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)
            task = cfg["experiment"]["task"]

            if task == "feature_only":
                out = model(batch["features"].float().to(device), y=None)
            elif task == "image_only":
                out = model(batch["image"].float().to(device), y=None)
            elif task == "image_feature":
                out = model(
                    batch["image"].float().to(device),
                    batch["features"].float().to(device),
                    y=None,
                )
            elif task == "phrase_only":
                out = model(batch["phrase_emb"].float().to(device), y=None)
            elif task == "image_phrase":
                out = model(
                    batch["image"].float().to(device),
                    batch["phrase_emb"].float().to(device),
                    y=None,
                )
            else:
                out = model(
                    image=batch["image"].float().to(device),
                    features=batch["features"].float().to(device),
                    phrase_emb=batch["phrase_emb"].float().to(device),
                    y=None,
                )

            preds.extend(out["pred"].cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

    return compute_metrics(
        targets,
        preds,
        num_classes=int(cfg["model"]["num_classes"]),
    )


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(int(cfg["train"].get("seed", 42)))

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, dl_train, dl_val = build_loaders(cfg)
    model = build_model(cfg, ds_train).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    best_qwk = -1e9

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch in tqdm(dl_train, desc=f"Epoch {epoch}"):
            out, loss = compute_batch_loss(cfg, model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = batch["y"].shape[0]
            total_loss += float(loss.item()) * bsz
            total_n += bsz

        val_metrics = evaluate(cfg, model, dl_val, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            **val_metrics,
        }
        print(epoch_result)

        save_json(epoch_result, output_dir / f"metrics_epoch_{epoch}.json")

        torch.save(
            {"model": model.state_dict(), "config": cfg, "epoch": epoch},
            output_dir / "last.pt",
        )

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch},
                output_dir / "best.pt",
            )

    print(f"Training finished. Best val QWK = {best_qwk:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
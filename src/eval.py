from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR))

from datasets.multimodal_us import DatasetConfig, MultimodalUSDataset
from train import build_model
from utils.metrics import compute_metrics, save_json


def main(config_path: str, checkpoint_path: str, split: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    split_file = Path(cfg["paths"]["split_dir"]) / f"{split}_ids.txt"

    ds = MultimodalUSDataset(
        DatasetConfig(
            task=cfg["experiment"]["task"],
            data_root=cfg["paths"]["data_root"],
            split_file=str(split_file),
            labels_csv=cfg["paths"].get("labels_csv"),
            feature_xlsx=cfg["paths"].get("feature_xlsx"),
            phrase_xlsx=cfg["paths"].get("phrase_xlsx"),
            image_size=cfg.get("model", {}).get("image_size", 384),
        )
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = build_model(cfg, ds)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ids, preds, targets = [], [], []

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
                    y=None
                )
            elif task == "phrase_only":
                out = model(batch["phrase_emb"].float().to(device), y=None)
            elif task == "image_phrase":
                out = model(
                    batch["image"].float().to(device),
                    batch["phrase_emb"].float().to(device),
                    y=None
                )
            else:
                out = model(
                    image=batch["image"].float().to(device),
                    features=batch["features"].float().to(device),
                    phrase_emb=batch["phrase_emb"].float().to(device),
                    y=None,
                )

            ids.extend(batch["id"])
            preds.extend(out["pred"].cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

    metrics = compute_metrics(targets, preds, num_classes=int(cfg["model"]["num_classes"]))

    run_dir = Path(checkpoint_path).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / f"{split}_predictions.csv"
    pred_df = pd.DataFrame({
        "id": ids,
        "y_true": targets,
        "y_pred": preds,
    })
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics_path = run_dir / f"{split}_metrics.json"
    save_json(metrics, metrics_path)

    print(metrics)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)
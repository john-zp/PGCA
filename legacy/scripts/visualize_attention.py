from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from datasets.multimodal_us import DatasetConfig, MultimodalUSDataset
from train import build_model


LABEL_INT_TO_NAME = {
    1: "2",
    2: "3",
    3: "4A",
    4: "4B",
    5: "4C",
    6: "5",
}


def denorm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-8)


def overlay_heatmap_on_gray(
    gray01: np.ndarray,
    heat01: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "jet",
) -> np.ndarray:
    gray01 = np.clip(gray01, 0.0, 1.0)
    heat01 = np.clip(heat01, 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat01)[..., :3]
    gray_rgb = np.stack([gray01, gray01, gray01], axis=-1)

    out = (1.0 - alpha) * gray_rgb + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return out


def roi_boundary(mask01: np.ndarray) -> np.ndarray:
    mask = (mask01 > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    up = np.pad(mask[:-1, :], ((1, 0), (0, 0)), mode="constant")
    down = np.pad(mask[1:, :], ((0, 1), (0, 0)), mode="constant")
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)), mode="constant")
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)), mode="constant")

    eroded = up & down & left & right
    boundary = mask ^ eroded
    return boundary.astype(np.uint8)


def draw_roi_on_gray(gray01: np.ndarray, roi01: np.ndarray) -> np.ndarray:
    gray_rgb = np.stack([gray01, gray01, gray01], axis=-1)
    bnd = roi_boundary(roi01)
    if bnd.sum() > 0:
        gray_rgb[bnd > 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.clip(gray_rgb, 0.0, 1.0)


def draw_roi_on_overlay(overlay_rgb: np.ndarray, roi01: np.ndarray) -> np.ndarray:
    out = overlay_rgb.copy()
    bnd = roi_boundary(roi01)
    if bnd.sum() > 0:
        out[bnd > 0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def attention_iou(heat01: np.ndarray, roi01: np.ndarray, thresh: float = 0.5) -> float:
    pred = (heat01 >= thresh).astype(np.uint8)
    gt = (roi01 >= 0.5).astype(np.uint8)
    union = (pred | gt).sum()
    if union == 0:
        return float("nan")
    inter = (pred & gt).sum()
    return float(inter / union)


def build_dataset_from_config(cfg: dict, split: str) -> MultimodalUSDataset:
    split_file = Path(cfg["paths"]["split_dir"]) / f"{split}_ids.txt"
    return MultimodalUSDataset(
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


def forward_one(model, batch: dict, task: str, device: torch.device) -> dict:
    if task == "image_phrase":
        out = model(
            batch["image"].float().to(device),
            batch["phrase_emb"].float().to(device),
            y=None,
        )
    elif task == "all_input":
        out = model(
            image=batch["image"].float().to(device),
            features=batch["features"].float().to(device),
            phrase_emb=batch["phrase_emb"].float().to(device),
            y=None,
        )
    else:
        raise ValueError(
            f"Task '{task}' is not supported by this visualization script. "
            "Use 'all_input' or 'image_phrase'."
        )
    return out


def save_single_views(
    out_dir: Path,
    sid: str,
    image01: np.ndarray,
    roi01: np.ndarray,
    heat01: np.ndarray,
    overlay_rgb: np.ndarray,
):
    plt.imsave(out_dir / f"{sid}_image.png", image01, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{sid}_roi.png", draw_roi_on_gray(image01, roi01))
    plt.imsave(out_dir / f"{sid}_heatmap.png", heat01, cmap="jet", vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{sid}_overlay.png", overlay_rgb)


def save_panel(
    out_dir: Path,
    sid: str,
    image01: np.ndarray,
    roi01: np.ndarray,
    heat01: np.ndarray,
    overlay_rgb: np.ndarray,
    gt_name: str,
    pred_name: str,
    iou50: float,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image01, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ultrasound image")
    axes[0].axis("off")

    axes[1].imshow(draw_roi_on_gray(image01, roi01))
    axes[1].set_title("Expert ROI")
    axes[1].axis("off")

    axes[2].imshow(heat01, cmap="jet", vmin=0.0, vmax=1.0)
    axes[2].set_title("Attention heatmap")
    axes[2].axis("off")

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    title = f"ID: {sid} | GT: {gt_name} | Pred: {pred_name} | IoU@0.5: {iou50:.3f}" if np.isfinite(iou50) else f"ID: {sid} | GT: {gt_name} | Pred: {pred_name} | IoU@0.5: NA"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml, e.g. configs/all_input.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="visualizations/all_input_test")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to export")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the chosen split")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    task = cfg["experiment"]["task"]
    if task not in {"all_input", "image_phrase"}:
        raise ValueError(
            f"This script only supports 'all_input' and 'image_phrase', got '{task}'."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset_from_config(cfg, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg, dataset)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    summary_rows = []
    end_index = min(len(dataset), args.start_index + args.num_samples)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx < args.start_index:
                continue
            if idx >= end_index:
                break

            out = forward_one(model, batch, task, device)

            sid = batch["id"][0] if isinstance(batch["id"], list) else batch["id"]
            sid = str(sid)

            y_true = int(batch["y"].item())
            y_pred = int(out["pred"].item())
            gt_name = LABEL_INT_TO_NAME.get(y_true, str(y_true))
            pred_name = LABEL_INT_TO_NAME.get(y_pred, str(y_pred))

            image = batch["image"][0].cpu().numpy()
            if image.ndim == 3:
                image = image[0]
            image01 = denorm01(image)

            if "roi_mask" in batch:
                roi = batch["roi_mask"][0].cpu().numpy()
                if roi.ndim == 3:
                    roi = roi[0]
                roi01 = (roi > 0.5).astype(np.float32)
            else:
                roi01 = np.zeros_like(image01, dtype=np.float32)

            attn = out["attn_map"]
            if attn.ndim == 4:
                attn = attn.squeeze(1)
            attn = attn[0]

            grid_hw = out["grid_hw"]
            if isinstance(grid_hw, (tuple, list)):
                h, w = int(grid_hw[0]), int(grid_hw[1])
            else:
                raise ValueError("Model output 'grid_hw' is missing or invalid.")

            heat_small = attn.view(h, w).detach().cpu()[None, None, ...]
            heat_up = F.interpolate(
                heat_small,
                size=image01.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            heat01 = denorm01(heat_up)

            overlay_rgb = overlay_heatmap_on_gray(image01, heat01, alpha=args.alpha)
            overlay_rgb = draw_roi_on_overlay(overlay_rgb, roi01)

            iou50 = attention_iou(heat01, roi01, thresh=0.5)

            save_single_views(out_dir, sid, image01, roi01, heat01, overlay_rgb)
            save_panel(out_dir, sid, image01, roi01, heat01, overlay_rgb, gt_name, pred_name, iou50)

            summary_rows.append({
                "id": sid,
                "y_true": y_true,
                "y_true_name": gt_name,
                "y_pred": y_pred,
                "y_pred_name": pred_name,
                "iou_at_0.5": iou50,
            })

            print(f"[{idx}] saved visualization for {sid}")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "y_true", "y_true_name", "y_pred", "y_pred_name", "iou_at_0.5"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved {len(summary_rows)} cases to: {out_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
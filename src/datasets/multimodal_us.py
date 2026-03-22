from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.io import (
    find_image_path,
    find_roi_path,
    load_grayscale_image,
    load_labels_csv,
    make_resized_roi_mask,
    read_feature_table,
    read_ids,
    read_phrase_table,
    parse_roi_txt,
)


@dataclass
class DatasetConfig:
    task: str
    data_root: str
    split_file: str
    labels_csv: Optional[str] = None
    feature_xlsx: Optional[str] = None
    phrase_xlsx: Optional[str] = None
    image_size: int = 384


class MultimodalUSDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        self.images_dir = self.data_root / "images"
        self.rois_dir = self.data_root / "rois"
        self.ids = read_ids(cfg.split_file)
        self.task = cfg.task
        self.image_size = cfg.image_size

        self.feature_df = None
        self.phrase_df = None
        self.label_map: dict[str, int] | None = None
        self.feature_cols: list[str] = []
        self.phrase_dim: int | None = None

        labels_csv = cfg.labels_csv or str(self.data_root / "labels.csv")
        if Path(labels_csv).exists():
            self.label_map = load_labels_csv(labels_csv)

        if cfg.feature_xlsx:
            self.feature_df = read_feature_table(cfg.feature_xlsx).set_index("id")
            self.feature_cols = [c for c in self.feature_df.columns if c not in ["BI-RADS", "label_int", "原始短语文本", "emb"]]
        if cfg.phrase_xlsx:
            self.phrase_df = read_phrase_table(cfg.phrase_xlsx).set_index("id")
            if len(self.phrase_df) > 0:
                self.phrase_dim = int(self.phrase_df.iloc[0]["emb_array"].shape[0])

        self.samples = []
        for sid in self.ids:
            record = {"id": sid}
            if self.feature_df is not None:
                if sid not in self.feature_df.index:
                    continue
                row = self.feature_df.loc[sid]
                if hasattr(row, "ndim") and getattr(row, "ndim", 1) > 1:
                    row = row.iloc[0]
                record["label_int"] = int(row["label_int"])
            if self.phrase_df is not None:
                if sid not in self.phrase_df.index:
                    continue
                row = self.phrase_df.loc[sid]
                if hasattr(row, "ndim") and getattr(row, "ndim", 1) > 1:
                    row = row.iloc[0]
                record["label_int"] = int(row["label_int"])
            if "label_int" not in record:
                if self.label_map is not None and sid in self.label_map:
                    record["label_int"] = int(self.label_map[sid])
                else:
                    continue
            if self.task in {"image_only", "image_feature", "image_phrase", "all_input"}:
                image_path = find_image_path(self.images_dir, sid)
                if image_path is None:
                    continue
                record["image_path"] = str(image_path)
            self.samples.append(record)

        if not self.samples:
            raise RuntimeError(
                f"No samples matched task={self.task}. Please check split file and data paths."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sid = sample["id"]
        item = {
            "id": sid,
            "y": torch.tensor(sample["label_int"], dtype=torch.long),
        }

        if self.feature_df is not None:
            row = self.feature_df.loc[sid]
            if hasattr(row, "ndim") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            feat = row[self.feature_cols].to_numpy(dtype=np.float32)
            item["features"] = torch.from_numpy(feat)

        if self.phrase_df is not None:
            row = self.phrase_df.loc[sid]
            if hasattr(row, "ndim") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            emb = row["emb_array"].astype(np.float32)
            item["phrase_emb"] = torch.from_numpy(emb)
            item["raw_text"] = str(row.get("原始短语文本", ""))

        if self.task in {"image_only", "image_feature", "image_phrase", "all_input"}:
            image_path = Path(sample["image_path"])
            img0 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img0 is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            orig_h, orig_w = img0.shape[:2]
            img = cv2.resize(img0, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
            lo, hi = np.percentile(img, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
            item["image"] = torch.from_numpy(img[None, ...])

            if self.task == "all_input":
                roi_path = find_roi_path(self.rois_dir, self.images_dir, sid)
                roi_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                roi_valid = 0.0
                if roi_path is not None:
                    roi = parse_roi_txt(roi_path)
                    if roi is not None:
                        roi_mask = make_resized_roi_mask(roi, (orig_h, orig_w), self.image_size)
                        roi_valid = 1.0
                item["roi_mask"] = torch.from_numpy(roi_mask[None, ...])
                item["roi_valid"] = torch.tensor(roi_valid, dtype=torch.float32)

        return item

"""Build a new, immutable cohort without modifying any historical artifact."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_pgca36_pathology_20261013"
SOURCE = ROOT / "data_remediation_20261013/original_materials_E"
HISTORY = ROOT / "data_remediation_20261013"
B = ROOT / "github_project/data/us"
LABELS = ["2", "3", "4A", "4B", "4C", "5"]
# These names are the audited option columns, not a positional crop of 38 inputs.
FEATURES = ["椭圆形", "圆形", "不规则形", "平行(纵横比<1)", "不平行(纵横比≥1)",
    "光整", "不光整", "模糊", "成角", "微小分叶", "毛刺", "无回声", "低回声", "等回声", "高回声",
    "囊实性回声(包括内有囊性变)", "不均匀回声", "无改变", "增强", "声影(衰减)", "混合性改变", "无钙化",
    "粗大钙化", "微钙化", "肿块内钙化", "肿块外钙化", "导管内钙化", "无血供", "内部血供", "边缘血供",
    "无相关特征", "结构扭曲", "导管改变", "皮肤增厚", "皮肤缩回", "水肿"]
SIZE = 224
DIRECT_TARGET = re.compile(r"BI[\s_-]*RADS|乳腺影像报告.{0,20}(?:分级|分类)|目标标签\s*[:：=]\s*(?:[1-6]|4[A-C])", re.I)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def key(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def write_json(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    path.chmod(0o600)


def write_csv(name, records, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (OUT / name).chmod(0o600)


class Union:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def load_form(filename):
    table = pd.read_excel(SOURCE / filename, header=None)
    choices = list(map(str, table.iloc[1].tolist()))
    assert choices[3:5] == ["良性", "恶性"] and choices[5:41] == FEATURES
    records = []
    for index, row in table.iloc[2:].iterrows():
        exam = key(row.iloc[0])
        if not exam or not exam.isdigit():
            continue
        values = pd.to_numeric(row.iloc[5:41], errors="coerce").to_numpy(dtype=float)
        targets = pd.to_numeric(row.iloc[41:49], errors="coerce").to_numpy(dtype=float)
        selected = np.flatnonzero(targets == 1)
        grade = choices[41 + selected[0]].replace("BI-RADS", "").strip().upper() if len(selected) == 1 else ""
        records.append({"exam": exam, "directory": key(row.iloc[2]), "pathology": key(row.iloc[1]),
            "features": values, "grade": grade, "file": filename, "row": int(index) + 1})
    return records


def parse_embedding(cell):
    array = np.asarray(ast.literal_eval(cell) if isinstance(cell, str) else cell, dtype=np.float32)
    if array.shape == (1, 512):
        array = array[0]
    if array.shape != (512,) or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
        raise ValueError("invalid_shape_nonfinite_or_zero_embedding")
    return array


def rectangle_mask(annotation, shape, convention):
    numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", annotation)]
    if len(numbers) != 4:
        return None
    x, y, a, b = numbers
    if convention == "center_xywh":
        x1, y1, x2, y2 = x - a / 2, y - b / 2, x + a / 2, y + b / 2
    elif convention == "top_left_xywh":
        x1, y1, x2, y2 = x, y, x + a, y + b
    else:
        x1, y1, x2, y2 = x, y, a, b
    h, w = shape
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    if x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    mask[top:bottom, left:right] = 1
    return mask if mask.any() else None


def roi_for(sid, shape):
    text_path = B / "rois" / (sid + ".txt")
    segmentation_path = SOURCE / "labelBinary" / (sid + ".jpg")
    if not text_path.is_file() or not segmentation_path.is_file():
        return np.zeros((SIZE, SIZE), np.uint8), {"roi_status": "unverifiable_missing_annotation_or_geometry_reference", "roi_convention": "", "roi_bbox_iou": ""}
    segmentation = cv2.imread(str(segmentation_path), cv2.IMREAD_GRAYSCALE)
    if segmentation is None or segmentation.shape != shape or not (segmentation > 127).any():
        return np.zeros((SIZE, SIZE), np.uint8), {"roi_status": "invalid_geometry_reference", "roi_convention": "", "roi_bbox_iou": ""}
    ys, xs = np.where(segmentation > 127)
    bounding = np.zeros(shape, np.uint8)
    bounding[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    candidates = []
    for convention in ["center_xywh", "top_left_xywh", "xyxy"]:
        mask = rectangle_mask(text_path.read_text(), shape, convention)
        if mask is not None:
            union = np.logical_or(mask, bounding).sum()
            overlap = np.logical_and(mask, bounding).sum() / union
            candidates.append((float(overlap), convention, mask))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    # Fixed geometric gates, not selected by label or predictive performance.
    if not candidates or candidates[0][0] < 0.80 or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.20):
        return np.zeros((SIZE, SIZE), np.uint8), {"roi_status": "ambiguous_coordinate_format", "roi_convention": "", "roi_bbox_iou": candidates[0][0] if candidates else ""}
    overlap, convention, mask = candidates[0]
    resized = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return resized, {"roi_status": "geometry_validated_rectangle", "roi_convention": convention, "roi_bbox_iou": overlap}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "data/dataset_manifest.csv").exists():
        raise FileExistsError("Cohort already locked; do not rebuild the split")
    (OUT / "data").mkdir(exist_ok=True)
    forms = load_form("bi_rads_internal_case_answers_formated2.xlsx") + load_form("bi_rads_internal_case_answers_2nd_par1t.xlsx")
    form_index = defaultdict(list)
    for record in forms:
        for identity in {record["directory"], record["exam"]} - {""}:
            form_index[identity].append(record)
    labels = pd.read_excel(SOURCE / "labels.xlsx", sheet_name="labels", dtype=str).fillna("")
    assert labels["id"].is_unique
    table = pd.read_excel(B / "X_with_all.xlsx")
    assert table["id"].is_unique
    allinput = table.set_index("id")
    history = pd.read_csv(HISTORY / "clean_dataset_manifest.csv", dtype=str, keep_default_na=False).set_index("image_id")
    assert set(labels["id"]) == set(history.index)
    records, exclusions, images, masks, features, embeddings = [], [], [], [], [], []
    for _, label_row in labels.sort_values("id").iterrows():
        sid, directory, grade = label_row["id"], key(label_row["图片子目录名称"]), key(label_row["BI-RADS"]).upper()
        reasons = []
        candidates = form_index[directory]
        if len(candidates) != 1:
            reasons.append("nonunique_or_missing_case_mapping")
        form = candidates[0] if len(candidates) == 1 else None
        old = history.loc[sid]
        if sid != old["exact_duplicate_representative"]:
            reasons.append("exact_duplicate_nonrepresentative")
        if grade not in LABELS or form is None or form["grade"] not in LABELS:
            reasons.append("illegal_or_out_of_scope_label")
        elif grade != form["grade"] or grade != old["primary_birads"] or grade != old["csv_birads"]:
            reasons.append("label_source_conflict")
        if form and (not np.isfinite(form["features"]).all() or not np.isin(form["features"], [0, 1]).all()):
            reasons.append("missing_or_nonbinary_observation")
        image_path = B / "images" / (sid + ".jpg")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            reasons.append("unreadable_image")
        elif sha(image_path.read_bytes()) != old["image_file_sha256"]:
            reasons.append("image_hash_changed_since_audit")
        embedding = None
        text = ""
        if sid not in allinput.index:
            reasons.append("missing_case_embedding_row")
        else:
            entry = allinput.loc[sid]
            text = key(entry["原始短语文本"])
            try:
                embedding = parse_embedding(entry["emb"])
            except (ValueError, SyntaxError, TypeError):
                reasons.append("invalid_case_embedding")
            if not form or not form["pathology"] or form["pathology"] not in text:
                reasons.append("pathology_text_source_not_confirmed")
            if DIRECT_TARGET.search(text):
                reasons.append("explicit_birads_target_in_text")
            if key(entry["BI-RADS"]).upper() != grade:
                reasons.append("embedding_table_label_conflict")
        if reasons:
            exclusions.append({"image_id": sid, "source_directory": directory, "reasons": "|".join(sorted(set(reasons)))})
            continue
        roi, roi_info = roi_for(sid, image.shape)
        resized = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        low, high = np.percentile(resized, [1, 99])
        resized = np.clip((resized - low) / max(float(high - low), 1.0), 0, 1)
        records.append({"image_id": sid, "array_index": len(records), "source_directory": directory,
            "patient_key": "P_" + sha(directory.encode())[:24], "group_id": old["conservative_group_id"],
            "patient_key_basis": "author_attestation_20261013_one_source_image_directory_per_patient",
            "birads": grade, "y": LABELS.index(grade) + 1, "source_form": form["file"], "source_excel_row": form["row"],
            "image_path": str(image_path), "image_sha256": old["image_file_sha256"],
            "pathology_text_sha256": sha(text.encode()), "embedding_sha256": sha(embedding.tobytes()),
            "embedding_source": str(B / "X_with_all.xlsx"), **roi_info})
        images.append(resized[None].astype(np.float16))
        masks.append(roi[None])
        features.append(form["features"].astype(np.float32))
        embeddings.append(embedding)
    if not records:
        raise ValueError("No valid common multimodal cohort")
    # Prior components already include source/check associations, raw-export aliases,
    # exact duplicates, and predeclared high-similarity edges. Verify all anew below.
    graph = Union([record["image_id"] for record in records])
    seen = {}
    for record in records:
        for identity in ["patient:" + record["patient_key"], "risk:" + record["group_id"]]:
            if identity in seen:
                graph.join(record["image_id"], seen[identity])
            else:
                seen[identity] = record["image_id"]
    components = defaultdict(list)
    for record in records:
        components[graph.find(record["image_id"])].append(record["image_id"])
    group_of = {sid: "G_" + sha("\n".join(sorted(ids)).encode())[:24] for ids in components.values() for sid in ids}
    y = np.array([record["y"] for record in records])
    groups = np.array([group_of[record["image_id"]] for record in records])
    split_indices = {}
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    for fold, (_, positions) in enumerate(splitter.split(np.zeros(len(records)), y, groups)):
        name = "train" if fold < 14 else "val" if fold < 17 else "test"
        for position in positions:
            split_indices[int(position)] = name
    patient_splits, group_splits, id_splits = defaultdict(set), defaultdict(set), {}
    for position, record in enumerate(records):
        record["group_id"], record["split"], record["split_seed"] = groups[position], split_indices[position], 42
        patient_splits[record["patient_key"]].add(record["split"])
        group_splits[record["group_id"]].add(record["split"])
        id_splits[record["image_id"]] = record["split"]
    assert all(len(values) == 1 for values in patient_splits.values())
    assert all(len(values) == 1 for values in group_splits.values())
    for filename in ["image_relationship_edges.csv", "high_similarity_pairs.csv"]:
        pairs = pd.read_csv(HISTORY / filename, dtype=str, keep_default_na=False)
        for _, pair in pairs.iterrows():
            if pair["image_a"] in id_splits and pair["image_b"] in id_splits:
                assert id_splits[pair["image_a"]] == id_splits[pair["image_b"]]
    write_csv("data/dataset_manifest.csv", records, list(records[0]))
    write_csv("split_manifest.csv", [{key: record[key] for key in ["image_id", "patient_key", "group_id", "birads", "y", "split", "split_seed"]} for record in records], ["image_id", "patient_key", "group_id", "birads", "y", "split", "split_seed"])
    write_csv("exclusions.csv", exclusions, ["image_id", "source_directory", "reasons"])
    for name, array in {"images": images, "roi_masks": masks, "features36": features, "pathology_embeddings": embeddings, "y": y}.items():
        np.save(OUT / "data" / (name + ".npy"), np.asarray(array), allow_pickle=False)
    schema = {"columns": FEATURES, "dimension": 36, "encoding": "original_selected_option_binary_indicators",
              "missing_policy": "isolate_not_zero_fill", "global_fitted_preprocessing": "none", "image_size": SIZE,
              "label_order": LABELS, "label_encoding": [1, 2, 3, 4, 5, 6], "text_mode": "pathology_known_at_training_and_inference"}
    write_json("feature_schema.json", schema)
    provenance_files = [SOURCE / name for name in ["bi_rads_internal_case_answers_formated2.xlsx", "bi_rads_internal_case_answers_2nd_par1t.xlsx", "labels.xlsx"]] + [B / "X_with_all.xlsx", HISTORY / "clean_dataset_manifest.csv", HISTORY / "image_relationship_edges.csv", HISTORY / "extra_original_export_alias_edges.csv"]
    hashes = {str(path): sha(path.read_bytes()) for path in provenance_files}
    generated = [path for path in (OUT / "data").iterdir() if path.is_file()] + [OUT / "split_manifest.csv", OUT / "feature_schema.json"]
    summary = {"status": "DATA_PREFLIGHT_PASSED", "images_input": len(labels), "images_eligible": len(records),
        "images_excluded": len(exclusions), "excluded_reasons": dict(Counter(reason for row in exclusions for reason in row["reasons"].split("|"))),
        "author_attested_patient_keys": len(patient_splits), "conservative_patient_risk_groups": len(group_splits),
        "grouping_independent_hospital_certification": False, "patient_assertion_source": "user_explicit_confirmation_in_current_task",
        "split_sizes": dict(Counter(row["split"] for row in records)),
        "split_class_counts": {name: dict(Counter(row["birads"] for row in records if row["split"] == name)) for name in ["train", "val", "test"]},
        "roi_status_counts": dict(Counter(row["roi_status"] for row in records)), "roi_convention_counts": dict(Counter(row["roi_convention"] for row in records)),
        "known_risk_edge_cross_split_count": 0, "source_hashes": hashes,
        "locked_file_hashes": {str(path.relative_to(OUT)): sha(path.read_bytes()) for path in generated},
        "old_quarantine_manifest_modified": False, "preprocessing_fit_on_test": False, "embeddings_generated": False}
    write_json("data_preflight.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

# 04_detect_characterize.py — processing script


#!/usr/bin/env python
"""
04_detect_characterize.py
=========================
Extract bounding‑boxes, centroids and volumes from nnU‑Net masks.

Outputs
-------
data/features/<subject_id>_bbox.json
data/features/tumor_info.json
"""

import json
import os
from pathlib import Path
import argparse
import SimpleITK as sitk
import numpy as np

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
from utils.config import (
    SEGMENTATION_DIR,
    FEATURES_DIR,
    DATA_INDEX_FILE,
)

LABELS = {
    "whole_tumor": {"value": (1, 2, 4)},  # union of BraTS labels
    "tumor_core":  {"value": (1, 4)},     # necrotic + enhancing
    "enhancing":   {"value": (4,)},
}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def load_seg(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))

def get_binary_mask(seg_img: sitk.Image, label_values):
    """Return binary mask for one or several label integers."""
    arr = sitk.GetArrayFromImage(seg_img)
    mask = np.isin(arr, label_values).astype(np.uint8)
    return sitk.GetImageFromArray(mask, isVector=False)

def stats_from_mask(mask_img: sitk.Image):
    """Use LabelShapeStatistics to get bbox, centroid, volume."""
    mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)

    label = 1  # single‑label binary mask
    bbox_vox = stats.GetBoundingBox(label)  # (x, y, z, sizeX, sizeY, sizeZ)
    centroid_vox = stats.GetCentroid(label)
    vol_vox = stats.GetPhysicalSize(label)  # in voxel‑count units

    spacing = np.array(mask_img.GetSpacing())
    # Bounding‑box: convert to physical (mm) [min,max] for each axis
    bbox_mm = [
        bbox_vox[0] * spacing[0],
        (bbox_vox[0] + bbox_vox[3]) * spacing[0],
        bbox_vox[1] * spacing[1],
        (bbox_vox[1] + bbox_vox[4]) * spacing[1],
        bbox_vox[2] * spacing[2],
        (bbox_vox[2] + bbox_vox[5]) * spacing[2],
    ]
    centroid_mm = np.multiply(centroid_vox, spacing).tolist()
    volume_mm3 = vol_vox * np.prod(spacing)

    return {
        "bounding_box_vox": list(bbox_vox),
        "bounding_box_mm":  [round(v, 1) for v in bbox_mm],
        "centroid_vox":     [round(c, 1) for c in centroid_vox],
        "centroid_mm":      [round(c, 1) for c in centroid_mm],
        "volume_mm3":       round(volume_mm3, 1),
    }

# -------------------------------------------------------------------------
# Main per‑subject processing
# -------------------------------------------------------------------------
def process_subject(subj_id: str, seg_path: Path) -> dict:
    seg_img = load_seg(seg_path)
    subject_info = {"subject_id": subj_id, "spacing": list(seg_img.GetSpacing())}

    for region, meta in LABELS.items():
        mask = get_binary_mask(seg_img, meta["value"])
        if sitk.GetArrayFromImage(mask).any():
            subject_info[region] = stats_from_mask(mask)
        else:
            subject_info[region] = None  # region not present

    return subject_info

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if JSON exists")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_INDEX_FILE) as f:
        subjects = json.load(f)

    all_info = []
    for subj_id in subjects:
        seg_path = Path(SEGMENTATION_DIR) / f"{subj_id}_seg.nii.gz"
        out_json = FEATURES_DIR / f"{subj_id}_bbox.json"

        if out_json.exists() and not args.overwrite:
            print(f"  {out_json.name} exists, skipping.")
            with open(out_json) as fj:
                all_info.append(json.load(fj))
            continue
        if not seg_path.exists():
            print(f"  Missing segmentation for {subj_id}, skip.")
            continue

        print(f" Detecting regions for {subj_id}")
        info = process_subject(subj_id, seg_path)
        with open(out_json, "w") as f:
            json.dump(info, f, indent=2)
        all_info.append(info)
        print(f" Saved {out_json}")

    # Aggregate
    tumor_info_path = FEATURES_DIR / "tumor_info.json"
    with open(tumor_info_path, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f" Tumor info written to {tumor_info_path} ({len(all_info)} subjects)")

if __name__ == "__main__":
    main()

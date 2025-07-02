# 05_biomarkers.py — processing script

#!/usr/bin/env python
"""
05_biomarkers.py
=================
Extract radiomics biomarkers from each tumor region.

Outputs:
---------
- data/biomarkers/biomarkers.csv
"""

import os
import json
import csv
from pathlib import Path

import SimpleITK as sitk
from radiomics import featureextractor
from utils.config import (
    PREPROCESSED_DIR,
    SEGMENTATION_DIR,
    BIOMARKERS_DIR,
    DATA_INDEX_FILE,
)

# Region definitions
REGIONS = {
    "whole_tumor": (1, 2, 4),
    "tumor_core":  (1, 4),
    "enhancing":   (4,)
}

# Initialize PyRadiomics extractor
params = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'verbose': False
}
extractor = featureextractor.RadiomicsFeatureExtractor(params)

def create_binary_mask(seg_img, labels):
    arr = sitk.GetArrayFromImage(seg_img)
    mask_arr = ((arr[..., None] == labels).any(axis=-1)).astype("uint8")
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(seg_img)
    return mask_img

def extract_subject_features(subj_id, image_path, seg_path):
    image = sitk.ReadImage(str(image_path))
    seg = sitk.ReadImage(str(seg_path))
    all_feats = {"subject_id": subj_id}

    for region_name, label_vals in REGIONS.items():
        mask = create_binary_mask(seg, label_vals)
        try:
            feats = extractor.execute(image, mask)
            # Filter only first-order and shape features
            for k, v in feats.items():
                if k.startswith("original_shape") or k.startswith("original_firstorder"):
                    all_feats[f"{region_name}__{k}"] = v
        except Exception as e:
            print(f"  {subj_id} {region_name}: {e}")
            for k in range(10):  # Fill placeholders
                all_feats[f"{region_name}__feature{k}"] = None

    return all_feats

def main():
    BIOMARKERS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = BIOMARKERS_DIR / "biomarkers.csv"

    with open(DATA_INDEX_FILE) as f:
        index = json.load(f)

    all_data = []
    for subj_id in index:
        image_path = Path(PREPROCESSED_DIR) / subj_id / "T1ce_preprocessed.nii.gz"
        seg_path = Path(SEGMENTATION_DIR) / f"{subj_id}_seg.nii.gz"
        if not image_path.exists() or not seg_path.exists():
            print(f"  Skipping {subj_id}, missing files.")
            continue

        print(f"Extracting biomarkers for {subj_id}")
        feats = extract_subject_features(subj_id, image_path, seg_path)
        all_data.append(feats)

    if all_data:
        fieldnames = sorted(set().union(*[d.keys() for d in all_data]))
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print(f" Saved biomarkers to {out_csv}")
    else:
        print(" No valid subjects found.")

if __name__ == "__main__":
    main()

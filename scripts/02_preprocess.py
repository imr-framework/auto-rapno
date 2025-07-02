# 02_preprocess.py — processing script
# scripts/02_preprocess.py – N4 + Resample + Normalize


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import SimpleITK as sitk
from utils.config import RAW_DIR, PREPROCESSED_DIR, DATA_INDEX_FILE, RESAMPLE_SPACING

def load_nifti(path):
    return sitk.ReadImage(path)

def save_nifti(img, path):
    sitk.WriteImage(img, path)

def n4_bias_correction(img):
    mask = sitk.OtsuThreshold(img, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(img, mask)

def resample_image(img, spacing=(1.0, 1.0, 1.0)):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc)) 
        for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)

def normalize_image(img):
    arr = sitk.GetArrayFromImage(img).astype("float32")
    arr = (arr - arr.mean()) / (arr.std() + 1e-5)
    return sitk.GetImageFromArray(arr, isVector=False)

def preprocess_subject(subject_id, modalities):
    subj_out_dir = os.path.join(PREPROCESSED_DIR, subject_id)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    for mod, path in modalities.items():
        img = load_nifti(path)
        img = n4_bias_correction(img)
        img = resample_image(img, RESAMPLE_SPACING)
        img = normalize_image(img)
        out_path = os.path.join(subj_out_dir, f"{mod}_preprocessed.nii.gz")
        save_nifti(img, out_path)
        print(f" Saved: {out_path}")

def main():
    with open(DATA_INDEX_FILE) as f:
        index = json.load(f)

    for subject_id, modalities in index.items():
        print(f" Preprocessing {subject_id}")
        preprocess_subject(subject_id, modalities)

if __name__ == "__main__":
    main()
# 01_parse_raw.py — processing script

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from utils.config import RAW_DIR, DATA_INDEX_FILE

# List of expected modalities
MODALITIES = ["T1N", "T1C", "T2W", "T2F", "SEG"]

def find_modalities(subj_folder):
    found = {}
    for fname in os.listdir(subj_folder):
        fpath = os.path.join(subj_folder, fname)
        if not fname.endswith(".nii.gz"):
            continue
        for mod in MODALITIES:
            if mod.lower() in fname.lower():
                found[mod] = fpath
                break
    return found

def main():
    training_dir = os.path.join(RAW_DIR, "BraTS-PEDs2024_Training")
    print(f"Scanning: {training_dir}")
    index = {}
    if not os.path.isdir(training_dir):
        print(f"Training directory not found: {training_dir}")
        return

    subject_ids = sorted(os.listdir(training_dir))
    print(f"Found {len(subject_ids)} subjects in BraTS-PEDs2024_Training.")
    for subject_id in subject_ids:
        subj_path = os.path.join(training_dir, subject_id)
        if not os.path.isdir(subj_path):
            continue
        print(f"Processing: BraTS-PEDs2024_Training/{subject_id}")
        mod_paths = find_modalities(subj_path)

        missing = [m for m in MODALITIES if m not in mod_paths]
        if missing:
            print(f"  Skipping BraTS-PEDs2024_Training/{subject_id} (missing: {missing})")
            continue

        index[f"BraTS-PEDs2024_Training/{subject_id}"] = {
            "folder": "BraTS-PEDs2024_Training",
            "subject_id": subject_id,
            "modalities": mod_paths
        }

    # Save the index
    with open(DATA_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

    print(f" Saved data index to: {DATA_INDEX_FILE}")
    print(f" {len(index)} subjects indexed.")

if __name__ == "__main__":
    main()

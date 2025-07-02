# 03_segment.py — processing script

#!/usr/bin/env python
"""
03_segment.py
=============
Run 3‑D nnU‑Net inference on pre‑processed BRATS‑PED cases.

Usage
-----
python scripts/03_segment.py \
    --model_dir models/nnunet_model \
    --planner nnUNetPlannerV2_3D \
    --device cuda:0            # or cpu

Outputs
-------
Segmentation masks saved to:
data/segmentation/{subject_id}_seg.nii.gz
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from utils.config import (
    PREPROCESSED_DIR,
    SEGMENTATION_DIR,
    DATA_INDEX_FILE,
)

################################################################################
# Helper functions
################################################################################


def run_nnunet_inference(
    case_dir: Path,
    output_mask: Path,
    model_dir: Path,
    planner_name: str,
    folds: str,
    device: str,
):
    """
    Call the nnU‑Net CLI (v2) for a single case.

    Parameters
    ----------
    case_dir : Path
        Folder that contains *_preprocessed.nii.gz files for one subject.
    output_mask : Path
        Destination path for the prediction NIfTI.
    model_dir : Path
        Trained nnU‑Net model folder (starts with `nnUNetTrainerXXX__`).
    planner_name : str
        Planner e.g. "nnUNetPlannerV2_3D".
    folds : str
        Which folds to use ("all" for ensemble, or "0", "1", ...).
    device : str
        "cpu" or e.g. "cuda:0".
    """

    tmp = tempfile.TemporaryDirectory()
    tmp_output = Path(tmp.name) / "pred.nii.gz"

    # nnUNet v2 CLI call
    cmd = [
        "nnUNetv2_predict",
        "--input", str(case_dir),
        "--output", str(tmp_output),
        "--model", str(model_dir),
        "--planner", planner_name,
        "--folds", folds,
        "--device", device,
        "--disable_tta",  # remove if you want Test‑Time Augmentation
    ]

    print(" Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Move prediction to pipeline folder
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    tmp_output.rename(output_mask)
    tmp.cleanup()
    print(f" Saved: {output_mask}")


################################################################################
# Main workflow
################################################################################


def main():
    parser = argparse.ArgumentParser(description="nnU‑Net 3‑D segmentation")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/nnunet_model",
        help="Path to trained nnU‑Net model folder",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="nnUNetPlannerV2_3D",
        help="Planner name used during training",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help='"all" for ensemble or comma‑sep list e.g. "0,1,2"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='"cpu" or "cuda:0", "cuda:1", ...',
    )
    args = parser.parse_args()

    # Load subject list from data_index.json
    with open(DATA_INDEX_FILE) as f:
        index = json.load(f)

    for subject_id in index:
        case_dir = Path(PREPROCESSED_DIR) / subject_id
        if not case_dir.exists():
            print(f"  Missing preprocessed data for {subject_id}, skipping.")
            continue

        output_mask = Path(SEGMENTATION_DIR) / f"{subject_id}_seg.nii.gz"

        # Skip if already segmented
        if output_mask.exists():
            print(f" {output_mask.name} already exists, skipping.")
            continue

        run_nnunet_inference(
            case_dir=case_dir,
            output_mask=output_mask,
            model_dir=Path(args.model_dir),
            planner_name=args.planner,
            folds=args.folds,
            device=args.device,
        )


if __name__ == "__main__":
    main()
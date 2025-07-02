# import os

# base_path = "auto_rapno/"  # <-- existing parent folder

# structure = [
#     "agents/data_agent.py",
#     "agents/prep_agent.py",
#     "agents/seg_agent.py",
#     "agents/detect_agent.py",
#     "agents/char_agent.py",
#     "agents/biomarker_agent.py",
#     "agents/xai_agent.py",
#     "agents/report_agent.py",

#     "configs/pipeline_config.yaml",
#     "configs/model_params.yaml",

#     "core/nnunet_wrapper.py",
#     "core/swinunetr_model.py",
#     "core/classifier_xgb.py",
#     "core/biomarker_extractor.py",

#     "data/raw/.gitkeep",
#     "data/processed/.gitkeep",
#     "data/outputs/.gitkeep",
#     "data/reports/.gitkeep",

#     "notebooks/01_data_exploration.ipynb",
#     "notebooks/02_segmentation_debug.ipynb",
#     "notebooks/03_xai_visualization.ipynb",
#     "notebooks/04_textual_explanation.ipynb",

#     "scripts/run_full_pipeline.py",
#     "scripts/run_segmentation_only.py",
#     "scripts/run_reporting.py",
#     "scripts/run_radiomics_analysis.py",

#     "tests/test_seg_agent.py",
#     "tests/test_xai_agent.py",
#     "tests/test_report_format.py",

#     "templates/rapno_report_template.html",

#     "ui/app.py",
#     "ui/utils.py",

#     "README.md",
#     "requirements.txt",
#     "environment.yaml",
#     "LICENSE"
# ]

# for file in structure:
#     full_path = os.path.join(base_path, file)
#     os.makedirs(os.path.dirname(full_path), exist_ok=True)
#     with open(full_path, "w") as f:
#         if file.endswith(".py"):
#             f.write("# " + os.path.basename(file))
#         elif file.endswith(".md"):
#             f.write("# Project Overview")
#         elif file.endswith(".yaml") or file.endswith(".yml"):
#             f.write("# YAML config")
#         elif file.endswith(".ipynb"):
#             f.write("{\"cells\": [], \"metadata\": {}, \"nbformat\": 4, \"nbformat_minor\": 5}")
#         elif file.endswith(".html"):
#             f.write("<!-- HTML Template for RAPNO report -->")
#         elif file.endswith(".txt"):
#             f.write("# requirements.txt")
#         elif file.endswith(".LICENSE"):
#             f.write("MIT License")



# Run simple structure creation

import os

# Base directory (you can change it)
base_dir = "brats_ped_pipeline"

# Folder structure
folders = [
    "data/raw",
    "data/preprocessed",
    "data/segmentation",
    "data/features",
    "data/biomarkers",
    "data/xai_maps",
    "data/reports",
    "scripts",
    "models/nnunet_model",
    "models/custom",
    "utils",
    "notebooks"
]

# File structure
files = [
    "scripts/01_parse_raw.py",
    "scripts/02_preprocess.py",
    "scripts/03_segment.py",
    "scripts/04_detect_characterize.py",
    "scripts/05_biomarkers.py",
    "scripts/06_explain.py",
    "scripts/07_generate_report.py",

    "utils/io_utils.py",
    "utils/nii_helpers.py",
    "utils/viz_utils.py",
    "utils/config.py",

    "notebooks/visualize_subject.ipynb",
    "notebooks/check_radiomics.ipynb",

    "data_index.json",
    "requirements.txt",
    "README.md",
    "run_all.sh"
]

# Create folders
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)

# Create files with simple content
for file in files:
    file_path = os.path.join(base_dir, file)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        if file.endswith(".py"):
            f.write(f"# {os.path.basename(file)} — processing script\n")
        elif file.endswith(".ipynb"):
            f.write("{\"cells\": [], \"metadata\": {}, \"nbformat\": 4, \"nbformat_minor\": 5}")
        elif file.endswith(".sh"):
            f.write("#!/bin/bash\n# Run all pipeline steps\n")
        elif file.endswith(".json"):
            f.write("{\n  \"subject_001\": {\n    \"T1\": \"data/raw/subject_001/T1.nii.gz\"\n  }\n}")
        elif file.endswith(".txt"):
            f.write("nibabel\nnumpy\npandas\nscikit-image\nscikit-learn\nSimpleITK\n")
        elif file.endswith(".md"):
            f.write("# BRATS-PED Pipeline\n\nNon-agentic, script-based pipeline for pediatric brain tumor processing.\n")
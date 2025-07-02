# config.py — processing script

import os

# Base directory (edit if needed)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
RAW_DIR_TRAIN = os.path.join(RAW_DIR, "BraTS-PEDs2024_Training")
RAW_DIR_VALID = os.path.join(RAW_DIR, "BraTS_Validation_Data_backup")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data/preprocessed")
SEGMENTATION_DIR = os.path.join(BASE_DIR, "data/segmentation")
FEATURES_DIR = os.path.join(BASE_DIR, "data/features")
BIOMARKERS_DIR = os.path.join(BASE_DIR, "data/biomarkers")
XAI_DIR = os.path.join(BASE_DIR, "data/xai_maps")
REPORTS_DIR = os.path.join(BASE_DIR, "data/reports")

# Index file
DATA_INDEX_FILE = os.path.join(BASE_DIR, "data_index.json")

print(f" Base directory: {BASE_DIR}")
print(f" Raw data:      {RAW_DIR}")
print(f"   Train:       {RAW_DIR_TRAIN}")
print(f"   Valid:       {RAW_DIR_VALID}")
print(f" Preprocessed:  {PREPROCESSED_DIR}")
print(f" Segmentation:  {SEGMENTATION_DIR}")
print(f" Features:      {FEATURES_DIR}")
print(f" Biomarkers:    {BIOMARKERS_DIR}")
print(f" XAI maps:      {XAI_DIR}")
print(f" Reports:       {REPORTS_DIR}")
print(f" Data index:    {DATA_INDEX_FILE}")

# Preprocessing settings
RESAMPLE_SPACING = (1.0, 1.0, 1.0)
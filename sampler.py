"""
=============================================================
ONE-TIME SAMPLER — CIC-DDoS2019
=============================================================
Run this ONCE to pre-save a small representative sample.
After this, your main script loads from sample.csv instantly.

Usage:
    python sampler.py
=============================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURATION
# =============================================================

DATA_DIR      = "/lclhome/mjawa009/LlmDal/cic ddos 2019/archive (1)"
OUTPUT_DIR    = "./outputs"
SAMPLE_OUT    = os.path.join(OUTPUT_DIR, "sample.csv")

N_PER_CLASS   = 500        # How many rows to sample per attack class
N_BENIGN      = 500        # How many benign rows to sample
RANDOM_STATE  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# LOAD ALL CSVs WITH PROGRESS
# =============================================================

csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

print(f"[INFO] Found {len(csv_files)} CSV files.\n")

dfs = []
for i, f in enumerate(csv_files, 1):
    print(f"[INFO] Reading {i}/{len(csv_files)}: {os.path.basename(f)}")
    dfs.append(pd.read_csv(f, low_memory=False))

print("\n[INFO] Concatenating...")
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()
print(f"[INFO] Total records loaded: {len(df):,}")

# =============================================================
# FIND LABEL COLUMN
# =============================================================

label_col = next(
    (c for c in df.columns if c.strip().lower() == "label"), None
)
if label_col is None:
    raise ValueError("Label column not found.")

df[label_col] = df[label_col].str.strip()
print(f"\n[INFO] Class distribution (full dataset):")
print(df[label_col].value_counts())

# =============================================================
# CLEAN
# =============================================================

df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n[INFO] Records after cleaning: {len(df):,}")

# =============================================================
# SAMPLE PER CLASS
# =============================================================

sampled = []

# Benign
benign = df[df[label_col].str.upper() == "BENIGN"]
benign_sample = benign.sample(
    n=min(N_BENIGN, len(benign)), random_state=RANDOM_STATE
)
sampled.append(benign_sample)
print(f"[INFO] Sampled {len(benign_sample)} BENIGN rows.")

# Each attack class
attack_classes = df[df[label_col].str.upper() != "BENIGN"][label_col].unique()
for cls in attack_classes:
    cls_df = df[df[label_col] == cls]
    cls_sample = cls_df.sample(
        n=min(N_PER_CLASS, len(cls_df)), random_state=RANDOM_STATE
    )
    sampled.append(cls_sample)
    print(f"[INFO] Sampled {len(cls_sample)} rows for class: {cls}")

# =============================================================
# SAVE
# =============================================================

final = pd.concat(sampled, ignore_index=True)
final.to_csv(SAMPLE_OUT, index=False)

print(f"\n[DONE] Sample saved to: {SAMPLE_OUT}")
print(f"[DONE] Total rows: {len(final):,}")
print(f"[DONE] From now on, point your main script to: {SAMPLE_OUT}")
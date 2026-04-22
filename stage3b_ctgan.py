"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 3b: CTGAN-Based Conditional Flow Generation
=============================================================

Goal:
    Train a CTGAN model on real attack + benign flows from
    sample.csv, then generate synthetic flows that blend both
    distributions to evade NIDS ML classifiers.

    Compare output against Stage 3a (LLM-based) in Stage 4.

Install:
    pip install ctgan

Outputs: ./outputs/synthetic_flows_ctgan.csv
=============================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # use only GPU device 0

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from ctgan import CTGAN


# =============================================================
# CONFIGURATION
# =============================================================

SAMPLE_PATH   = "./outputs/sample.csv"
OUTPUT_DIR    = "./outputs"
OUTPUT_FILE   = "synthetic_flows_ctgan.csv"

# Number of synthetic rows to generate
N_SYNTHETIC   = 1000     # increase freely — CTGAN is fast

# CTGAN training epochs (more = better quality, slower)
EPOCHS        = 300

# Features to use — same 19 as Stage 3a for fair comparison
SELECTED_FEATURES = [
    "Source Port",
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
]

# Label column name
LABEL_COL = "Label"

RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# STEP 1 — LOAD AND PREPARE DATA
# =============================================================

def load_data():
    print(f"[INFO] Loading sample data from: {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    # Verify features exist
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")

    print(f"[INFO] Total records loaded: {len(df):,}")
    print(f"[INFO] Label distribution:\n{df[LABEL_COL].value_counts()}\n")

    # Keep only selected features + label
    df = df[SELECTED_FEATURES + [LABEL_COL]]

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[INFO] Records after cleaning: {len(df):,}\n")
    return df


# =============================================================
# STEP 2 — BUILD BLENDED TRAINING SET
# =============================================================

def build_training_set(df):
    """
    Combine attack and benign flows into a single training set.
    CTGAN will learn the joint distribution of both, producing
    synthetic flows that sit between the two distributions —
    preserving attack-like structure while shifting toward benign
    ranges to reduce classifier detectability.
    """
    attack_df = df[df[LABEL_COL].str.upper() != "BENIGN"]
    benign_df = df[df[LABEL_COL].str.upper() == "BENIGN"]

    print(f"[INFO] Attack flows:  {len(attack_df):,}")
    print(f"[INFO] Benign flows:  {len(benign_df):,}")

    # Use all available data — CTGAN benefits from more samples
    train_df = pd.concat([attack_df, benign_df], ignore_index=True)
    train_df = train_df[SELECTED_FEATURES]  # drop label for generation

    print(f"[INFO] Training set size: {len(train_df):,} rows x {len(SELECTED_FEATURES)} features\n")
    return train_df


# =============================================================
# STEP 3 — TRAIN CTGAN
# =============================================================

def train_ctgan(train_df):
    print(f"[INFO] Training CTGAN for {EPOCHS} epochs...")
    print("[INFO] This takes 2-5 minutes on CPU, under 1 min on GPU.\n")

    model = CTGAN(
        epochs=EPOCHS,
        batch_size=500,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        verbose=True,
    )

    # No discrete columns — all features are numeric
    model.fit(train_df, discrete_columns=[])

    print("\n[INFO] CTGAN training complete.\n")
    return model


# =============================================================
# STEP 4 — GENERATE SYNTHETIC FLOWS
# =============================================================

def generate_flows(model, n):
    print(f"[INFO] Generating {n:,} synthetic flows...")
    synthetic = model.sample(n)
    print(f"[INFO] Generated shape: {synthetic.shape}\n")
    return synthetic


# =============================================================
# STEP 5 — VALIDATE AND CLIP
# =============================================================

def validate_and_clip(synthetic_df, reference_df):
    """
    Post-process synthetic flows:
    - Clip values to the real data range (prevents out-of-range anomalies)
    - Drop any remaining NaNs
    - Enforce non-negative values for packet/byte features
    """
    print("[INFO] Validating and clipping synthetic flows...")

    for col in SELECTED_FEATURES:
        real_min = reference_df[col].min()
        real_max = reference_df[col].max()
        synthetic_df[col] = synthetic_df[col].clip(lower=real_min, upper=real_max)

    # All network flow values should be >= 0
    synthetic_df = synthetic_df.clip(lower=0)
    synthetic_df = synthetic_df.dropna()

    before = len(synthetic_df)
    synthetic_df = synthetic_df.drop_duplicates()
    dupes_removed = before - len(synthetic_df)

    print(f"[INFO] Duplicates removed: {dupes_removed}")
    print(f"[INFO] Final synthetic flow count: {len(synthetic_df):,}\n")
    return synthetic_df.reset_index(drop=True)


# =============================================================
# MAIN PIPELINE
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  STAGE 3b — CTGAN FLOW GENERATION")
    print("="*55 + "\n")

    # Step 1 — Load
    df = load_data()

    # Step 2 — Build training set
    train_df = build_training_set(df)

    # Step 3 — Train CTGAN
    model = train_ctgan(train_df)

    # Step 4 — Generate
    synthetic_raw = generate_flows(model, N_SYNTHETIC)

    # Step 5 — Validate
    reference_df = train_df  # use training data as reference for clipping
    synthetic_df = validate_and_clip(synthetic_raw, reference_df)

    # Save
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    synthetic_df.to_csv(out_path, index=False)

    print(f"[DONE] {len(synthetic_df):,} synthetic flows saved to: {out_path}")
    print(f"       Features: {list(synthetic_df.columns)}")
    print("\n[DONE] Stage 3b complete.")
    print("       Compare against Stage 3a (LLM) output in Stage 4.\n")
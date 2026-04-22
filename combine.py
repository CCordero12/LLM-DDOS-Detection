"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 3c: Combine Real + Synthetic Data for Classifier Training
=============================================================

What this does:
    1. Loads the real traffic from sample.csv
    2. Loads synthetic flows from Stage 3a (LLM) and/or Stage 3b (CTGAN)
    3. Aligns columns so both share the same feature set
    4. Combines them into one training-ready CSV
    5. Keeps a held-out real-only test set so evaluation metrics
       reflect real-world performance, not synthetic data quality

Output files:
    ./outputs/combined_train.csv   — real + synthetic, for training
    ./outputs/real_test.csv        — real data only, for evaluation

Why combine:
    The goal is to inflate the DDoS attack class so the classifier
    is exposed to more attack variety during training. Training on
    synthetic-only data would give misleadingly high accuracy because
    the model would just learn the synthetic distribution, not real
    attack patterns.

Usage:
    python stage3c_combine.py

Requirements:
    sample.csv                     must exist in ./outputs/
    synthetic_flows.csv            output of stage3_llm_generation.py
    synthetic_flows_ctgan.csv      output of stage3b_ctgan.py
    (at least one synthetic file must be present)
=============================================================
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


# =============================================================
# CONFIGURATION
# =============================================================

OUTPUT_DIR         = "./outputs"
SAMPLE_PATH        = "./outputs/sample.csv"
SYNTH_LLM_PATH     = "./outputs/synthetic_flows.csv"
SYNTH_CTGAN_PATH   = "./outputs/synthetic_flows_ctgan.csv"

TRAIN_LLM_OUTPUT_PATH   = "./outputs/combined_train_llm.csv"
TRAIN_CTGAN_OUTPUT_PATH = "./outputs/combined_train_ctgan.csv"
TEST_OUTPUT_PATH        = "./outputs/real_test.csv"

# Fraction of REAL data held out as a clean test set.
# Generated ONCE and shared by both LLM and CTGAN runs for fair comparison.
TEST_SIZE          = 0.2
RANDOM_STATE       = 42

LABEL_COL          = "Label"


# =============================================================
# STEP 1 — LOAD REAL DATA
# =============================================================

def load_real(path):
    print(f"[INFO] Loading real data: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop pandas index artifacts
    unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"[INFO] Dropped index artifact columns: {unnamed}")

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError("No 'Label' column found in sample.csv.")

    df[label_col] = df[label_col].str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"[INFO] Real rows loaded   : {len(df):,}")
    print(f"[INFO] Real label distribution:\n{df[label_col].value_counts().to_string()}\n")
    return df


# =============================================================
# STEP 2 — LOAD SYNTHETIC DATA
# =============================================================

def load_synthetic(llm_path, ctgan_path, use_llm, use_ctgan):
    frames = []

    if use_llm and llm_path:
        if os.path.exists(llm_path):
            df = pd.read_csv(llm_path, low_memory=False)
            df.columns = df.columns.str.strip()
            unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
            if unnamed:
                df = df.drop(columns=unnamed)
            if LABEL_COL not in df.columns:
                print(f"[WARN] LLM file has no Label column — generation still running. Skipping Run A.")
            else:
                print(f"[INFO] LLM synthetic rows   : {len(df):,}")
                print(f"[INFO] LLM labels:\n{df[LABEL_COL].value_counts().to_string()}\n")
                frames.append(df)
        else:
            print(f"[WARN] LLM synthetic file not found: {llm_path} — skipping.")

    if use_ctgan and ctgan_path:
        if os.path.exists(ctgan_path):
            df = pd.read_csv(ctgan_path, low_memory=False)
            df.columns = df.columns.str.strip()
            unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
            if unnamed:
                df = df.drop(columns=unnamed)
            if LABEL_COL not in df.columns:
                print(f"[WARN] CTGAN file has no Label column — skipping.")
            else:
                print(f"[INFO] CTGAN synthetic rows : {len(df):,}")
                print(f"[INFO] CTGAN labels:\n{df[LABEL_COL].value_counts().to_string()}\n")
                frames.append(df)
        else:
            print(f"[WARN] CTGAN synthetic file not found: {ctgan_path} — skipping.")

    if not frames:
        raise FileNotFoundError(
            "No synthetic files found with a Label column. "
            "Check that generation scripts have finished."
        )

    synthetic_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total synthetic rows : {len(synthetic_df):,}")
    print(f"[INFO] Combined synthetic labels:\n{synthetic_df[LABEL_COL].value_counts().to_string()}\n")
    return synthetic_df


# =============================================================
# STEP 3 — ALIGN COLUMNS
# =============================================================

def align_columns(real_df, synth_df):
    """
    Align synthetic columns to the numeric features of real data only.
    String columns (Flow ID, Source IP, Timestamp etc.) are dropped
    from real_df FIRST so median fills never touch non-numeric data.
    """
    # Step 1: restrict real_df to numeric columns only
    numeric_real  = real_df.select_dtypes(include=[np.number]).columns.tolist()
    real_features = [c for c in numeric_real if c != LABEL_COL]
    real_df       = real_df[real_features + [LABEL_COL]].copy()

    # Step 2: drop synthetic columns not in real numeric features
    synth_features = [c for c in synth_df.columns if c != LABEL_COL]
    extra  = set(synth_features) - set(real_features)
    missing = set(real_features) - set(synth_features)

    if extra:
        print(f"[INFO] Dropping {len(extra)} synthetic-only columns.")
        synth_df = synth_df.drop(columns=[c for c in extra if c in synth_df.columns])

    # Step 3: fill missing features using real numeric medians (safe now)
    if missing:
        print(f"[INFO] {len(missing)} features missing from synthetic — filling with real median.")
        for col in missing:
            synth_df[col] = real_df[col].median()

    # Step 4: reorder synthetic columns to match real exactly
    synth_df = synth_df[real_features + [LABEL_COL]]

    print(f"[INFO] Aligned feature count: {len(real_features)}\n")
    return real_df, synth_df, real_features


# =============================================================
# STEP 4 — SPLIT REAL DATA INTO TRAIN / TEST
# =============================================================

def split_real(real_df):
    """
    Hold out TEST_SIZE of real data as a clean test set.
    Synthetic data is NEVER added to the test set — this ensures
    evaluation metrics reflect real-world detection performance.
    """
    from sklearn.preprocessing import LabelEncoder
    le_temp = LabelEncoder()
    y_temp  = le_temp.fit_transform(real_df[LABEL_COL])

    # Check minimum class size for stratified split
    counts    = np.bincount(y_temp)
    min_count = counts.min()
    min_needed = max(2, int(np.ceil(1 / TEST_SIZE)))

    if min_count < min_needed:
        print(f"[WARN] Smallest class has {min_count} rows — falling back to non-stratified split.")
        train_real, test_real = train_test_split(
            real_df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        train_real, test_real = train_test_split(
            real_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=real_df[LABEL_COL]
        )

    print(f"[INFO] Real train rows : {len(train_real):,}")
    print(f"[INFO] Real test rows  : {len(test_real):,}  (held out — no synthetic data here)\n")
    return train_real, test_real


# =============================================================
# STEP 5 — COMBINE AND SAVE ONE TRAINING SET
# =============================================================

def combine_and_save(train_real, synth_df, out_path, label):
    combined = pd.concat([train_real, synth_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    combined.to_csv(out_path, index=False)

    n_total  = len(combined)
    n_attack = (combined[LABEL_COL].str.upper() != "BENIGN").sum()
    n_benign = (combined[LABEL_COL].str.upper() == "BENIGN").sum()

    print(f"[INFO] {label} training set: {n_total:,} rows")
    print(f"       Label distribution:")
    print(combined[LABEL_COL].value_counts().to_string())
    print(f"       Attack rate: {n_attack/n_total*100:.1f}%  |  Benign rate: {n_benign/n_total*100:.1f}%")
    print(f"       Saved to: {out_path}\n")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  STAGE 3c — COMBINE REAL + SYNTHETIC DATA")
    print("  (LLM and CTGAN evaluated separately)")
    print("="*55 + "\n")

    # Step 1 — load real data
    real_df = load_real(SAMPLE_PATH)

    # Step 2 — split real ONCE into train/test
    # The SAME test set is used for both LLM and CTGAN evaluation
    # so the comparison is fair — same held-out real rows every time.
    train_real, test_real = split_real(real_df)
    test_real.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"[INFO] real_test.csv saved : {len(test_real):,} rows")
    print(f"       This is your evaluation set — use it for BOTH LLM and CTGAN comparison.\n")

    # Step 3 — RUN A: Real + LLM synthetic
    print("=" * 55)
    print("  RUN A — Real + LLM synthetic")
    print("=" * 55)
    llm_ok = False
    if os.path.exists(SYNTH_LLM_PATH):
        df_check = pd.read_csv(SYNTH_LLM_PATH, nrows=1)
        df_check.columns = df_check.columns.str.strip()
        if LABEL_COL in df_check.columns:
            llm_df = pd.read_csv(SYNTH_LLM_PATH, low_memory=False)
            llm_df.columns = llm_df.columns.str.strip()
            unnamed = [c for c in llm_df.columns if c.strip().lower().startswith("unnamed")]
            if unnamed:
                llm_df = llm_df.drop(columns=unnamed)
            print(f"[INFO] LLM synthetic rows: {len(llm_df):,}")
            print(f"[INFO] LLM labels:\n{llm_df[LABEL_COL].value_counts().to_string()}\n")
            _, llm_df, _ = align_columns(real_df, llm_df)
            combine_and_save(train_real, llm_df, TRAIN_LLM_OUTPUT_PATH, "LLM")
            llm_ok = True
        else:
            print("[WARN] LLM file has no Label column — generation still running. Skipping Run A.\n")
    else:
        print(f"[WARN] LLM synthetic not found at {SYNTH_LLM_PATH} — skipping Run A.\n")
    if not llm_ok:
        print("[INFO] Run A skipped — re-run combine.py after LLM generation finishes.\n")

    # Step 4 — RUN B: Real + CTGAN synthetic
    print("=" * 55)
    print("  RUN B — Real + CTGAN synthetic")
    print("=" * 55)
    ctgan_ok = False
    if os.path.exists(SYNTH_CTGAN_PATH):
        df_check = pd.read_csv(SYNTH_CTGAN_PATH, nrows=1)
        df_check.columns = df_check.columns.str.strip()
        if LABEL_COL in df_check.columns:
            ctgan_df = pd.read_csv(SYNTH_CTGAN_PATH, low_memory=False)
            ctgan_df.columns = ctgan_df.columns.str.strip()
            unnamed = [c for c in ctgan_df.columns if c.strip().lower().startswith("unnamed")]
            if unnamed:
                ctgan_df = ctgan_df.drop(columns=unnamed)
            print(f"[INFO] CTGAN synthetic rows: {len(ctgan_df):,}")
            print(f"[INFO] CTGAN labels:\n{ctgan_df[LABEL_COL].value_counts().to_string()}\n")
            _, ctgan_df, _ = align_columns(real_df, ctgan_df)
            combine_and_save(train_real, ctgan_df, TRAIN_CTGAN_OUTPUT_PATH, "CTGAN")
            ctgan_ok = True
        else:
            print("[WARN] CTGAN file has no Label column. Skipping Run B.\n")
    else:
        print(f"[WARN] CTGAN synthetic not found at {SYNTH_CTGAN_PATH} — skipping Run B.\n")
    if not ctgan_ok:
        print("[INFO] Run B skipped — check CTGAN output file.\n")

    print("=" * 55)
    print("[DONE] Files saved:")
    print(f"       {TEST_OUTPUT_PATH}          ← shared eval set for both")
    print(f"       {TRAIN_LLM_OUTPUT_PATH}     ← train with LLM synthetic")
    print(f"       {TRAIN_CTGAN_OUTPUT_PATH}   ← train with CTGAN synthetic")
    print()
    print("  NEXT STEP:")
    print("  Train classifier on combined_train_llm.csv   → evaluate on real_test.csv")
    print("  Train classifier on combined_train_ctgan.csv → evaluate on real_test.csv")
    print("  Compare detection rate, FAR, F1 between the two.")
    print("=" * 55 + "\n")
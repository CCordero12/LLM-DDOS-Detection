"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 1: Data Loading & Preprocessing
Stage 2: Train Random Forest & XGBoost Classifiers
=============================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    roc_auc_score, top_k_accuracy_score
)
from xgboost import XGBClassifier


# =============================================================
# CONFIGURATION — edit these paths to match your setup
# =============================================================

DATA_DIR = r"C:\Users\Admin\Documents\Cybersecurity_final\data"           # Folder containing CIC-DDoS2019 CSV files
OUTPUT_DIR = r"C:\Users\Admin\Documents\Cybersecurity_final\outputs"      # Folder to save models and results
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Max rows kept per label class across the whole dataset.
# 200,000 per class × 11 classes ≈ 2.2M rows total → ~3–4 GB RAM peak.
# Raise to 500_000 if you want more data and have headroom.
ROWS_PER_CLASS = 200_000

# Rows read from disk per CSV chunk. 100k is safe for 16 GB RAM.
CHUNKSIZE = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# =============================================================
# STAGE 1 — DATA LOADING & PREPROCESSING
# Two-pass streaming: no full DataFrame ever held in memory.
# =============================================================

# Columns to drop regardless of dataset
DROP_COLS = {"Flow ID", "Source IP", "Destination IP",
             "Source Port", "Destination Port", "Timestamp"}


def get_csv_files(data_dir):
    """Recursively find all CSVs and group by immediate parent folder."""
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    csv_files += glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}' or its subdirectories.\n"
            f"Expected structure:\n"
            f"  {data_dir}\\\n"
            f"    01-12\\  <-- Day 1 CSVs here\n"
            f"    03-11\\  <-- Day 2 CSVs here"
        )

    folder_map = {}
    for f in csv_files:
        folder = os.path.basename(os.path.dirname(f))
        folder_map.setdefault(folder, []).append(f)

    print(f"[INFO] Found {len(csv_files)} CSV file(s) across all subdirectories:")
    for folder, files in sorted(folder_map.items()):
        print(f"  [{folder}]")
        for f in files:
            print(f"    - {os.path.basename(f)}")

    return csv_files, folder_map


def stream_and_preprocess(data_dir):
    """
    Two-pass streaming pipeline — never holds more than one chunk in RAM.

    PASS 1: Scan every file to discover:
            - Shared feature column names
            - All class labels (to fit LabelEncoder)
            - Per-column 99th-pct caps (to clip inf values)
            - Label distribution per folder

    PASS 2: Re-stream every file, clean each chunk on the fly,
            reservoir-sample up to ROWS_PER_CLASS rows per label,
            and accumulate only the final numpy arrays.
    """

    csv_files, folder_map = get_csv_files(data_dir)

    # ------------------------------------------------------------------ #
    # PASS 1 — discover schema, labels, column caps                       #
    # ------------------------------------------------------------------ #
    print("\n[PASS 1] Scanning files for schema, labels, and inf caps...")

    all_labels          = set()
    col_stats           = {}   # col -> list of per-chunk 99th-pct values
    feature_cols        = None
    label_col           = None
    folder_label_counts = {}   # folder -> {label: count}

    for folder, files in sorted(folder_map.items()):
        folder_label_counts[folder] = {}
        for f in files:
            fname = os.path.basename(f)
            try:
                reader = pd.read_csv(f, low_memory=False,
                                     chunksize=CHUNKSIZE, on_bad_lines="skip")
                for chunk in reader:
                    chunk.columns = chunk.columns.str.strip()

                    # Detect label column once
                    if label_col is None:
                        label_col = next(
                            (c for c in chunk.columns
                             if c.strip().lower() == "label"), None
                        )

                    if label_col and label_col in chunk.columns:
                        chunk[label_col] = chunk[label_col].str.strip()
                        for lbl, cnt in chunk[label_col].value_counts().items():
                            all_labels.add(lbl)
                            folder_label_counts[folder][lbl] = (
                                folder_label_counts[folder].get(lbl, 0) + cnt
                            )

                    # Agree on feature columns from first valid chunk
                    if feature_cols is None and label_col:
                        feature_cols = [
                            c for c in chunk.columns
                            if c not in DROP_COLS
                            and c != label_col
                            and pd.api.types.is_numeric_dtype(chunk[c])
                        ]

                    # Accumulate per-column 99th pct for clipping
                    if feature_cols:
                        num_chunk = chunk[
                            [c for c in feature_cols if c in chunk.columns]
                        ].replace([np.inf, -np.inf], np.nan)
                        for col in num_chunk.columns:
                            p99 = num_chunk[col].quantile(0.99)
                            col_stats.setdefault(col, []).append(p99)

            except Exception as e:
                print(f"  [WARNING] Skipping {fname} in pass 1 — {e}")

    if label_col is None or feature_cols is None:
        raise ValueError("Could not detect label column or numeric features. "
                         "Check your CSV files.")

    # Per-column cap = median of per-chunk 99th percentiles
    col_caps = {col: float(np.nanmedian(vals))
                for col, vals in col_stats.items()}

    # Print per-folder label breakdown
    for folder, counts in folder_label_counts.items():
        print(f"\n  Label breakdown for [{folder}]:")
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<30} {cnt:>10,}")

    # Fit LabelEncoder on the full discovered label set
    le = LabelEncoder()
    le.fit(sorted(all_labels))
    print(f"\n[INFO] Classes found ({len(le.classes_)}): {list(le.classes_)}")
    print(f"[INFO] Feature count  : {len(feature_cols)}")
    print(f"[INFO] Rows-per-class : {ROWS_PER_CLASS:,}")

    # ------------------------------------------------------------------ #
    # PASS 2 — stream, clean, sample, accumulate numpy arrays             #
    # ------------------------------------------------------------------ #
    print("\n[PASS 2] Streaming, cleaning, and sampling data...")

    # reservoir[label] = list of numpy arrays (rows of features)
    reservoir        = {lbl: [] for lbl in le.classes_}
    reservoir_counts = {lbl: 0  for lbl in le.classes_}
    total_seen       = 0
    total_dropped    = 0

    for f in csv_files:
        fname = os.path.basename(f)
        try:
            reader = pd.read_csv(f, low_memory=False,
                                 chunksize=CHUNKSIZE, on_bad_lines="skip")
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()

                if label_col not in chunk.columns:
                    continue

                # Only keep columns we know about
                available = [c for c in feature_cols if c in chunk.columns]
                if not available:
                    continue

                chunk[label_col] = chunk[label_col].str.strip()
                before = len(chunk)
                total_seen += before

                # Clip inf → cap, drop NaN
                sub = chunk[[label_col] + available].copy()
                for col in available:
                    sub[col] = (
                        sub[col]
                        .replace([np.inf, -np.inf], np.nan)
                        .clip(upper=col_caps.get(col, np.nan))
                    )
                sub.dropna(inplace=True)
                total_dropped += before - len(sub)

                # Reservoir-sample per label
                for lbl, grp in sub.groupby(label_col):
                    if lbl not in reservoir:
                        continue
                    needed = ROWS_PER_CLASS - reservoir_counts[lbl]
                    if needed <= 0:
                        continue
                    rows = grp[available].values
                    if len(rows) > needed:
                        idx  = np.random.choice(len(rows), needed, replace=False)
                        rows = rows[idx]
                    reservoir[lbl].append(rows)
                    reservoir_counts[lbl] += len(rows)

        except Exception as e:
            print(f"  [WARNING] Skipping {fname} in pass 2 — {e}")

    print(f"[INFO] Total rows scanned : {total_seen:,}")
    print(f"[INFO] Rows dropped (NaN) : {total_dropped:,}")
    print(f"[INFO] Rows kept per class:")
    for lbl, cnt in reservoir_counts.items():
        print(f"    {lbl:<30} {cnt:>8,}")

    # Stack into final arrays
    X_list, y_list = [], []
    for lbl, arrays in reservoir.items():
        if not arrays:
            print(f"  [WARNING] No rows collected for class '{lbl}' — skipping.")
            continue
        X_block = np.vstack(arrays)
        y_block = np.full(len(X_block),
                          le.transform([lbl])[0], dtype=np.int64)
        X_list.append(X_block)
        y_list.append(y_block)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # Shuffle combined array
    perm  = np.random.permutation(len(X_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    print(f"\n[INFO] Final dataset shape : {X_all.shape}")

    # Scale
    print("[INFO] Fitting StandardScaler...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Save artifacts for Stage 3 / 4
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    pd.Series(feature_cols, name="feature").to_csv(
        os.path.join(OUTPUT_DIR, "feature_names.csv"), index=False
    )
    print("[INFO] Scaler, LabelEncoder, and feature names saved.")
    print(f"[INFO] Classes encoded : "
          f"{dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    return X_scaled, y_all, le, feature_cols


def split_data(X, y):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")
    return X_train, X_test, y_train, y_test


# =============================================================
# STAGE 2 — TRAIN CLASSIFIERS
# =============================================================

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    print("[INFO] Random Forest training complete.")
    return rf


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    print("[INFO] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    xgb.fit(X_train, y_train)
    print("[INFO] XGBoost training complete.")
    return xgb



def compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name):
    """
    Compute DDoS-specific detection metrics on top of standard ML metrics.

    Metrics reported
    ----------------
    Attack Traffic Rate   : % of test flows that are genuine attacks
    Detection Rate (DR)   : % of attack flows correctly identified (macro recall over attack classes)
    False Alarm Rate (FAR): % of benign flows wrongly classified as attacks
    Miss Rate             : % of attacks missed (1 - DR)
    Per-attack-class DR   : per-class recall for every attack type
    """
    benign_idx = list(le.classes_).index("BENIGN")
    n_total    = len(y_test)

    # Boolean masks
    is_attack_true  = (y_test != benign_idx)
    is_attack_pred  = (y_pred != benign_idx)
    is_benign_true  = ~is_attack_true

    # ── Global rates ──────────────────────────────────────────
    n_attacks_true  = is_attack_true.sum()
    n_benign_true   = is_benign_true.sum()

    attack_traffic_rate = n_attacks_true / n_total

    # Detection Rate — true positives over all real attacks
    tp_attacks = (is_attack_true & is_attack_pred).sum()
    dr = tp_attacks / n_attacks_true if n_attacks_true > 0 else 0.0

    # False Alarm Rate — benign flows flagged as attack
    fp_benign  = (is_benign_true & is_attack_pred).sum()
    far = fp_benign / n_benign_true if n_benign_true > 0 else 0.0

    miss_rate = 1.0 - dr

    # ── Per-attack-class detection rate ───────────────────────
    per_class_dr = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        if cls_idx == benign_idx:
            continue
        mask_true = (y_test == cls_idx)
        if mask_true.sum() == 0:
            continue
        mask_correct = mask_true & (y_pred == cls_idx)
        per_class_dr[cls_name] = mask_correct.sum() / mask_true.sum()

    # ── Print ─────────────────────────────────────────────────
    W = 55
    print(f"\n{'─'*W}")
    print(f"  {model_name} — DDoS Detection Metrics")
    print(f"{'─'*W}")
    print(f"  {'Attack Traffic Rate':<35} {attack_traffic_rate*100:>8.2f} %")
    print(f"  {'Detection Rate (DR)':<35} {dr*100:>8.2f} %")
    print(f"  {'False Alarm Rate (FAR)':<35} {far*100:>8.2f} %")
    print(f"  {'Miss Rate (1 - DR)':<35} {miss_rate*100:>8.2f} %")
    print(f"  {'─'*35} {'─'*8}")
    print(f"  Per-Attack-Class Detection Rate:")
    for cls_name, rate in sorted(per_class_dr.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"    {cls_name:<28} {rate*100:>6.2f}%  {bar}")
    print(f"{'─'*W}\n")

    # ── Save to CSV ───────────────────────────────────────────
    rows = [{
        "model": model_name,
        "attack_traffic_rate_%": round(attack_traffic_rate * 100, 4),
        "detection_rate_%": round(dr * 100, 4),
        "false_alarm_rate_%": round(far * 100, 4),
        "miss_rate_%": round(miss_rate * 100, 4),
    }]
    for cls_name, rate in per_class_dr.items():
        rows[0][f"dr_{cls_name}_%"] = round(rate * 100, 4)

    csv_path = os.path.join(
        OUTPUT_DIR, f"{model_name.replace(' ', '_')}_ddos_metrics.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  DDoS metrics CSV saved to: {csv_path}")

    return dr, far

def evaluate_model(model, X_test, y_test, model_name, le):
    """Evaluate a trained model with a full suite of metrics and save confusion matrix."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)
    n_classes   = len(le.classes_)

    # ── Core metrics ──────────────────────────────────────────
    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    prec_w   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1_w     = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    f1_mac   = f1_score(y_test, y_pred,        average="macro",    zero_division=0)
    f1_mic   = f1_score(y_test, y_pred,        average="micro",    zero_division=0)

    # ── Advanced metrics ───────────────────────────────────────
    mcc      = matthews_corrcoef(y_test, y_pred)
    kappa    = cohen_kappa_score(y_test, y_pred)
    ll       = log_loss(y_test, y_prob)
    top2     = top_k_accuracy_score(y_test, y_prob, k=min(2, n_classes))
    top3     = top_k_accuracy_score(y_test, y_prob, k=min(3, n_classes))

    # ROC-AUC (one-vs-rest, weighted) — needs at least 2 classes present in test
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    # ── Print report ───────────────────────────────────────────
    W = 55
    print(f"\n{'='*W}")
    print(f"  {model_name} — Full Metric Report")
    print(f"{'='*W}")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'Accuracy':<35} {acc:>10.4f}")
    print(f"  {'Balanced Accuracy':<35} {bal_acc:>10.4f}")
    print(f"  {'Top-2 Accuracy':<35} {top2:>10.4f}")
    print(f"  {'Top-3 Accuracy':<35} {top3:>10.4f}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'Weighted Precision':<35} {prec_w:>10.4f}")
    print(f"  {'Weighted Recall':<35} {rec_w:>10.4f}")
    print(f"  {'Weighted F1':<35} {f1_w:>10.4f}")
    print(f"  {'Macro F1':<35} {f1_mac:>10.4f}")
    print(f"  {'Micro F1':<35} {f1_mic:>10.4f}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'ROC-AUC (OvR, weighted)':<35} {auc:>10.4f}")
    print(f"  {'Log Loss':<35} {ll:>10.4f}")
    print(f"  {'Matthews Corr. Coef. (MCC)':<35} {mcc:>10.4f}")
    print(f"  {'Cohen\'s Kappa':<35} {kappa:>10.4f}")
    print(f"{'='*W}")
    print(f"\n  Per-Class Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, zero_division=0))

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(max(10, n_classes), max(7, n_classes)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plot_path = os.path.join(
        OUTPUT_DIR, f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {plot_path}")

    # ── Save metrics to CSV ────────────────────────────────────
    metrics_dict = {
        "model": model_name,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "top2_accuracy": top2, "top3_accuracy": top3,
        "weighted_precision": prec_w, "weighted_recall": rec_w,
        "weighted_f1": f1_w, "macro_f1": f1_mac, "micro_f1": f1_mic,
        "roc_auc_ovr_weighted": auc, "log_loss": ll,
        "mcc": mcc, "cohen_kappa": kappa,
    }
    metrics_path = os.path.join(
        OUTPUT_DIR, f"{model_name.replace(' ', '_')}_metrics.csv"
    )
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
    print(f"  Metrics CSV saved to    : {metrics_path}\n")

    # ── DDoS-specific detection metrics ───────────────────────
    compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name)

    return acc, f1_w


def save_model(model, filename):
    """Persist a trained model to disk."""
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")


# =============================================================
# MAIN PIPELINE
# =============================================================

if __name__ == "__main__":

    # --- Stage 1 ---
    print("\n" + "="*50)
    print("  STAGE 1 — DATA LOADING & PREPROCESSING")
    print("="*50 + "\n")

    X, y, le, feature_names = stream_and_preprocess(DATA_DIR)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Free the full arrays — only train/test splits needed from here
    del X, y

    # --- Stage 2 ---
    print("\n" + "="*50)
    print("  STAGE 2 — TRAINING CLASSIFIERS")
    print("="*50 + "\n")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest", le)
    save_model(rf_model, "random_forest.pkl")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost", le)
    save_model(xgb_model, "xgboost.pkl")

    print("\n[DONE] Stage 1 & 2 complete. Models and artifacts saved to ./outputs/")
    print("       Ready for Stage 3 — LLM-based conditional flow generation.\n")
"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 4: Train & Evaluate on Combined Data
=============================================================

Runs TWICE — once with LLM-augmented training data,
once with CTGAN-augmented training data — evaluating
both on the SAME real_test.csv held-out set.

This is the fair LLM vs CTGAN comparison.

Inputs:
    ./outputs/combined_train_llm.csv    real + LLM synthetic
    ./outputs/combined_train_ctgan.csv  real + CTGAN synthetic
    ./outputs/real_test.csv             held-out real data only

Outputs per run (suffixed with _llm or _ctgan):
    RF and XGBoost .pkl models
    confusion matrix PNGs
    metrics CSVs
    DDoS detection metrics CSVs

Usage:
    python stage4_train_eval.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

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
# CONFIGURATION
# =============================================================

OUTPUT_DIR             = "./outputs"
REAL_TEST_PATH         = "./outputs/real_test.csv"
TRAIN_LLM_PATH         = "./outputs/combined_train_llm.csv"
TRAIN_CTGAN_PATH       = "./outputs/combined_train_ctgan.csv"

LABEL_COL              = "Label"
RANDOM_STATE           = 42

# Non-numeric / identifier columns to drop before training
DROP_COLS = {"Flow ID", "Source IP", "Destination IP",
             "Timestamp", "SimillarHTTP", "Inbound"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# LOAD AND PREPROCESS A CSV INTO X, y
# =============================================================

def load_and_preprocess(csv_path, label_encoder=None, scaler=None, fit=False):
    """
    Load a CSV, drop non-numeric columns, encode labels, scale features.

    fit=True  → fit the LabelEncoder and StandardScaler on this data
                (use for training set)
    fit=False → transform using already-fitted objects
                (use for test set)

    Returns X (numpy), y (numpy), label_encoder, scaler, feature_names
    """
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop index artifacts
    unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Drop non-numeric identifier columns
    to_drop = [c for c in df.columns if c in DROP_COLS]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Find label column
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"No Label column found in {csv_path}")

    df[label_col] = df[label_col].str.strip()

    # Keep only numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != label_col]

    df = df[feature_cols + [label_col]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"[INFO] Rows: {len(df):,}  |  Features: {len(feature_cols)}")
    print(f"[INFO] Label distribution:\n{df[label_col].value_counts().to_string()}\n")

    X_raw = df[feature_cols].values
    y_raw = df[label_col].values

    if fit:
        label_encoder = LabelEncoder()
        label_encoder.fit(sorted(set(y_raw)))
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        # Test set: only transform, never fit
        # Handle any labels in test that weren't in train
        known = set(label_encoder.classes_)
        unknown = set(y_raw) - known
        if unknown:
            print(f"[WARN] Test labels not seen in training (will be dropped): {unknown}")
            mask = np.isin(y_raw, list(known))
            X_raw = X_raw[mask]
            y_raw = y_raw[mask]
            df    = df[mask]
        X = scaler.transform(X_raw)

    y = label_encoder.transform(y_raw)

    return X, y, label_encoder, scaler, feature_cols


# =============================================================
# TRAIN CLASSIFIERS
# =============================================================

def train_random_forest(X_train, y_train):
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    print("[INFO] Random Forest done.\n")
    return rf


def train_xgboost(X_train, y_train):
    print("[INFO] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)
    print("[INFO] XGBoost done.\n")
    return xgb


# =============================================================
# EVALUATE
# =============================================================

def compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name, suffix):
    benign_idx     = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else -1
    n_total        = len(y_test)
    is_attack_true = (y_test != benign_idx)
    is_attack_pred = (y_pred != benign_idx)
    is_benign_true = ~is_attack_true

    n_attacks_true      = is_attack_true.sum()
    n_benign_true       = is_benign_true.sum()
    attack_traffic_rate = n_attacks_true / n_total if n_total > 0 else 0

    tp_attacks = (is_attack_true & is_attack_pred).sum()
    dr  = tp_attacks / n_attacks_true if n_attacks_true > 0 else 0.0
    fp  = (is_benign_true & is_attack_pred).sum()
    far = fp / n_benign_true if n_benign_true > 0 else 0.0

    per_class_dr = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        if cls_idx == benign_idx:
            continue
        mask = (y_test == cls_idx)
        if mask.sum() == 0:
            continue
        per_class_dr[cls_name] = (mask & (y_pred == cls_idx)).sum() / mask.sum()

    W = 60
    print(f"\n{'─'*W}")
    print(f"  {model_name} [{suffix.upper()}] — DDoS Detection Metrics")
    print(f"{'─'*W}")
    print(f"  {'Attack Traffic Rate':<38} {attack_traffic_rate*100:>8.2f} %")
    print(f"  {'Detection Rate (DR)':<38} {dr*100:>8.2f} %")
    print(f"  {'False Alarm Rate (FAR)':<38} {far*100:>8.2f} %")
    print(f"  {'Miss Rate (1 - DR)':<38} {(1-dr)*100:>8.2f} %")
    print(f"  Per-Attack-Class Detection Rate:")
    for cls_name, rate in sorted(per_class_dr.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"    {cls_name:<30} {rate*100:>6.2f}%  {bar}")
    print(f"{'─'*W}\n")

    rows = [{
        "model": model_name,
        "augmentation": suffix,
        "attack_traffic_rate_%": round(attack_traffic_rate * 100, 4),
        "detection_rate_%":      round(dr  * 100, 4),
        "false_alarm_rate_%":    round(far * 100, 4),
        "miss_rate_%":           round((1 - dr) * 100, 4),
    }]
    for cls_name, rate in per_class_dr.items():
        rows[0][f"dr_{cls_name}_%"] = round(rate * 100, 4)

    csv_path = os.path.join(OUTPUT_DIR, f"{model_name.replace(' ','_')}_{suffix}_ddos_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  DDoS metrics saved to: {csv_path}")
    return dr, far


def evaluate_model(model, X_test, y_test, model_name, le, suffix):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    n_classes = len(le.classes_)

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_w  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w   = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1_w    = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    f1_mac  = f1_score(y_test, y_pred,        average="macro",    zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_pred)
    kappa   = cohen_kappa_score(y_test, y_pred)
    # Use all classes from the label encoder as the reference set.
    # Training may have more classes than test (e.g. ATTACK_SYNTHETIC only
    # appears in training, never in real_test.csv). Passing labels= explicitly
    # keeps the probability matrix aligned with the full class list.
    all_labels = list(range(len(le.classes_)))
    try:
        ll = log_loss(y_test, y_prob, labels=all_labels)
    except Exception:
        ll = float("nan")
    top2    = top_k_accuracy_score(y_test, y_prob, k=min(2, n_classes), labels=all_labels)
    top3    = top_k_accuracy_score(y_test, y_prob, k=min(3, n_classes), labels=all_labels)

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted",
                            labels=all_labels)
    except ValueError:
        auc = float("nan")

    W = 60
    print(f"\n{'='*W}")
    print(f"  {model_name} [{suffix.upper()}] — Metrics on real_test.csv")
    print(f"{'='*W}")
    print(f"  {'Accuracy':<38} {acc:>10.4f}")
    print(f"  {'Balanced Accuracy':<38} {bal_acc:>10.4f}")
    print(f"  {'Top-2 Accuracy':<38} {top2:>10.4f}")
    print(f"  {'Top-3 Accuracy':<38} {top3:>10.4f}")
    print(f"  {'Weighted Precision':<38} {prec_w:>10.4f}")
    print(f"  {'Weighted Recall':<38} {rec_w:>10.4f}")
    print(f"  {'Weighted F1':<38} {f1_w:>10.4f}")
    print(f"  {'Macro F1':<38} {f1_mac:>10.4f}")
    print(f"  {'ROC-AUC (OvR, weighted)':<38} {auc:>10.4f}")
    print(f"  {'Log Loss':<38} {ll:>10.4f}")
    print(f"  {'MCC':<38} {mcc:>10.4f}")
    print(f"  {'Cohen Kappa':<38} {kappa:>10.4f}")
    print(f"{'='*W}")
    print(f"\n  Per-Class Report:\n")
    # Use only classes actually present in test set for report/confusion matrix
    test_classes_idx  = sorted(set(y_test))
    test_classes_names = [le.classes_[i] for i in test_classes_idx]
    print(classification_report(y_test, y_pred,
                                labels=test_classes_idx,
                                target_names=test_classes_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=test_classes_idx)
    plt.figure(figsize=(max(10, len(test_classes_idx)), max(7, len(test_classes_idx))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_classes_names, yticklabels=test_classes_names)
    plt.title(f"{model_name} [{suffix}] — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR,
        f"{model_name.replace(' ','_')}_{suffix}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {plot_path}")

    # Metrics CSV
    metrics_dict = {
        "model": model_name, "augmentation": suffix,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "top2_accuracy": top2, "top3_accuracy": top3,
        "weighted_precision": prec_w, "weighted_recall": rec_w,
        "weighted_f1": f1_w, "macro_f1": f1_mac,
        "roc_auc_ovr_weighted": auc, "log_loss": ll,
        "mcc": mcc, "cohen_kappa": kappa,
    }
    metrics_path = os.path.join(OUTPUT_DIR,
        f"{model_name.replace(' ','_')}_{suffix}_metrics.csv")
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
    print(f"  Metrics saved to: {metrics_path}\n")

    compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name, suffix)
    return acc, f1_w


# =============================================================
# ONE FULL RUN (train on combined, eval on real test)
# =============================================================

def run(train_path, suffix):
    print("\n" + "="*60)
    print(f"  RUN: {suffix.upper()} — training on {os.path.basename(train_path)}")
    print(f"       evaluating on {os.path.basename(REAL_TEST_PATH)}")
    print("="*60 + "\n")

    # Load and fit on training data
    X_train, y_train, le, scaler, feature_names = load_and_preprocess(
        train_path, fit=True
    )

    # Load and transform test data using same le + scaler
    X_test, y_test, le, scaler, _ = load_and_preprocess(
        REAL_TEST_PATH, label_encoder=le, scaler=scaler, fit=False
    )

    print(f"[INFO] Train shape : {X_train.shape}")
    print(f"[INFO] Test shape  : {X_test.shape}\n")

    # Save scaler and label encoder for this run
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{suffix}.pkl"))
    joblib.dump(le,     os.path.join(OUTPUT_DIR, f"label_encoder_{suffix}.pkl"))

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest", le, suffix)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, f"random_forest_{suffix}.pkl"))

    # XGBoost
    xgb = train_xgboost(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, "XGBoost", le, suffix)
    joblib.dump(xgb, os.path.join(OUTPUT_DIR, f"xgboost_{suffix}.pkl"))

    print(f"[DONE] {suffix.upper()} run complete.\n")


# =============================================================
# MAIN — run both, compare
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  STAGE 4 — TRAIN & EVALUATE (LLM vs CTGAN)")
    print("="*60)

    results = []

    if os.path.exists(TRAIN_LLM_PATH):
        run(TRAIN_LLM_PATH, suffix="llm")
    else:
        print(f"[WARN] {TRAIN_LLM_PATH} not found — skipping LLM run.")

    if os.path.exists(TRAIN_CTGAN_PATH):
        run(TRAIN_CTGAN_PATH, suffix="ctgan")
    else:
        print(f"[WARN] {TRAIN_CTGAN_PATH} not found — skipping CTGAN run.")

    # Print side-by-side summary from saved CSVs
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    summary_rows = []
    for suffix in ["llm", "ctgan"]:
        for model_name in ["Random_Forest", "XGBoost"]:
            path = os.path.join(OUTPUT_DIR, f"{model_name}_{suffix}_ddos_metrics.csv")
            if os.path.exists(path):
                row = pd.read_csv(path).iloc[0].to_dict()
                summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ["model", "augmentation", "detection_rate_%",
                "false_alarm_rate_%", "miss_rate_%", "attack_traffic_rate_%"]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
        print(f"\n  Full summary saved to: ./outputs/comparison_summary.csv")

    print("\n[DONE] Stage 4 complete.\n")
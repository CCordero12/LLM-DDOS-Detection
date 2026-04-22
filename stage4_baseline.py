"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 4 Baseline: Train & Evaluate on Real Data Only
=============================================================

Trains RF and XGBoost on 80% of sample.csv (real data only,
no synthetic augmentation) and evaluates on real_test.csv.

This is the baseline to compare against LLM and CTGAN runs.

Expected output layout:
    Baseline (real only)       → DR, FAR, F1
    + LLM synthetic (stage4)   → DR, FAR, F1
    + CTGAN synthetic (stage4) → DR, FAR, F1

Inputs:
    ./outputs/sample.csv       real traffic
    ./outputs/real_test.csv    held-out test set (same one used
                               by stage4_train_eval.py)

Outputs:
    ./outputs/Random_Forest_baseline_metrics.csv
    ./outputs/XGBoost_baseline_metrics.csv
    ./outputs/Random_Forest_baseline_ddos_metrics.csv
    ./outputs/XGBoost_baseline_ddos_metrics.csv
    ./outputs/Random_Forest_baseline_confusion_matrix.png
    ./outputs/XGBoost_baseline_confusion_matrix.png

Usage:
    python stage4_baseline.py
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
# CONFIGURATION
# =============================================================

SAMPLE_PATH    = "./outputs/sample.csv"
REAL_TEST_PATH = "./outputs/real_test.csv"
OUTPUT_DIR     = "./outputs"
SUFFIX         = "baseline"

# Must match the same split used in combine.py so the test set
# is truly held out and never seen during baseline training.
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

DROP_COLS = {"Flow ID", "Source IP", "Destination IP",
             "Timestamp", "SimillarHTTP", "Inbound"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# LOAD AND PREPROCESS
# =============================================================

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    to_drop = [c for c in df.columns if c in DROP_COLS]
    if to_drop:
        df = df.drop(columns=to_drop)

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"No Label column in {path}")

    df[label_col] = df[label_col].str.strip()

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != label_col]

    df = df[feature_cols + [label_col]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"[INFO] {os.path.basename(path)}: {len(df):,} rows, {len(feature_cols)} features")
    print(f"[INFO] Labels:\n{df[label_col].value_counts().to_string()}\n")
    return df, feature_cols, label_col


# =============================================================
# METRICS
# =============================================================

def compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name):
    benign_idx     = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else -1
    n_total        = len(y_test)
    is_attack_true = (y_test != benign_idx)
    is_attack_pred = (y_pred != benign_idx)
    is_benign_true = ~is_attack_true

    n_attacks = is_attack_true.sum()
    n_benign  = is_benign_true.sum()
    atr       = n_attacks / n_total if n_total > 0 else 0

    tp  = (is_attack_true & is_attack_pred).sum()
    dr  = tp / n_attacks if n_attacks > 0 else 0.0
    fp  = (is_benign_true & is_attack_pred).sum()
    far = fp / n_benign if n_benign > 0 else 0.0

    per_class_dr = {}
    for idx, name in enumerate(le.classes_):
        if idx == benign_idx:
            continue
        mask = (y_test == idx)
        if mask.sum() == 0:
            continue
        per_class_dr[name] = (mask & (y_pred == idx)).sum() / mask.sum()

    W = 60
    print(f"\n{'─'*W}")
    print(f"  {model_name} [BASELINE] — DDoS Detection Metrics")
    print(f"{'─'*W}")
    print(f"  {'Attack Traffic Rate':<38} {atr*100:>8.2f} %")
    print(f"  {'Detection Rate (DR)':<38} {dr*100:>8.2f} %")
    print(f"  {'False Alarm Rate (FAR)':<38} {far*100:>8.2f} %")
    print(f"  {'Miss Rate (1 - DR)':<38} {(1-dr)*100:>8.2f} %")
    print(f"  Per-Attack-Class Detection Rate:")
    for name, rate in sorted(per_class_dr.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"    {name:<30} {rate*100:>6.2f}%  {bar}")
    print(f"{'─'*W}\n")

    rows = [{
        "model": model_name, "augmentation": SUFFIX,
        "attack_traffic_rate_%": round(atr * 100, 4),
        "detection_rate_%":      round(dr  * 100, 4),
        "false_alarm_rate_%":    round(far * 100, 4),
        "miss_rate_%":           round((1 - dr) * 100, 4),
    }]
    for name, rate in per_class_dr.items():
        rows[0][f"dr_{name}_%"] = round(rate * 100, 4)

    out = os.path.join(OUTPUT_DIR, f"{model_name.replace(' ','_')}_{SUFFIX}_ddos_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  DDoS metrics saved to: {out}")
    return dr, far


def evaluate_model(model, X_test, y_test, model_name, le):
    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    n_classes = len(le.classes_)
    all_labels = list(range(n_classes))

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_w  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w   = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1_w    = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    f1_mac  = f1_score(y_test, y_pred,        average="macro",    zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_pred)
    kappa   = cohen_kappa_score(y_test, y_pred)

    try:
        ll = log_loss(y_test, y_prob, labels=all_labels)
    except Exception:
        ll = float("nan")

    top2 = top_k_accuracy_score(y_test, y_prob, k=min(2, n_classes), labels=all_labels)
    top3 = top_k_accuracy_score(y_test, y_prob, k=min(3, n_classes), labels=all_labels)

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr",
                            average="weighted", labels=all_labels)
    except ValueError:
        auc = float("nan")

    W = 60
    print(f"\n{'='*W}")
    print(f"  {model_name} [BASELINE] — Metrics on real_test.csv")
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

    test_idx   = sorted(set(y_test))
    test_names = [le.classes_[i] for i in test_idx]
    print(f"\n  Per-Class Report:\n")
    print(classification_report(y_test, y_pred, labels=test_idx,
                                target_names=test_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=test_idx)
    plt.figure(figsize=(max(10, len(test_idx)), max(7, len(test_idx))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_names, yticklabels=test_names)
    plt.title(f"{model_name} [baseline] — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR,
        f"{model_name.replace(' ','_')}_{SUFFIX}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {plot_path}")

    metrics_dict = {
        "model": model_name, "augmentation": SUFFIX,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "top2_accuracy": top2, "top3_accuracy": top3,
        "weighted_precision": prec_w, "weighted_recall": rec_w,
        "weighted_f1": f1_w, "macro_f1": f1_mac,
        "roc_auc_ovr_weighted": auc, "log_loss": ll,
        "mcc": mcc, "cohen_kappa": kappa,
    }
    out = os.path.join(OUTPUT_DIR,
        f"{model_name.replace(' ','_')}_{SUFFIX}_metrics.csv")
    pd.DataFrame([metrics_dict]).to_csv(out, index=False)
    print(f"  Metrics saved to: {out}\n")

    compute_ddos_metrics(y_test, y_pred, y_prob, le, model_name)
    return acc, f1_w


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  STAGE 4 BASELINE — Real Data Only (no synthetic)")
    print("="*60 + "\n")

    # Load sample.csv and split into train/test using the SAME
    # random state and test_size as combine.py — this ensures
    # the baseline test split is identical to real_test.csv.
    print("[INFO] Loading sample.csv for baseline training split...")
    train_df, feature_cols, label_col = load_csv(SAMPLE_PATH)

    # Stratified split — same seed as combine.py
    train_real, _ = train_test_split(
        train_df, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[label_col]
    )
    print(f"[INFO] Baseline train rows: {len(train_real):,}  "
          f"(80% of sample.csv — no synthetic rows)\n")

    # Fit LabelEncoder and Scaler on training data
    le = LabelEncoder()
    le.fit(sorted(train_real[label_col].unique()))

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_real[feature_cols].values)
    y_train = le.transform(train_real[label_col].values)

    # Load real_test.csv and transform with same le + scaler
    print("[INFO] Loading real_test.csv for evaluation...")
    test_df, _, _ = load_csv(REAL_TEST_PATH)

    # Align test features to training feature set
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        print(f"[WARN] {len(missing)} features missing from test — filling with 0.")
        for c in missing:
            test_df[c] = 0.0
    test_df = test_df.reindex(columns=feature_cols + [label_col], fill_value=0.0)

    # Drop test rows with labels unseen in training
    known   = set(le.classes_)
    unknown = set(test_df[label_col]) - known
    if unknown:
        print(f"[WARN] Dropping {len(unknown)} unknown test labels: {unknown}")
        test_df = test_df[test_df[label_col].isin(known)]

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = le.transform(test_df[label_col].values)

    print(f"[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    # Save artifacts
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_baseline.pkl"))
    joblib.dump(le,     os.path.join(OUTPUT_DIR, "label_encoder_baseline.pkl"))

    # Random Forest
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    print("[INFO] Random Forest done.\n")
    evaluate_model(rf, X_test, y_test, "Random Forest", le)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest_baseline.pkl"))

    # XGBoost
    print("[INFO] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        tree_method="hist", eval_metric="mlogloss",
        n_jobs=-1, random_state=RANDOM_STATE, use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    print("[INFO] XGBoost done.\n")
    evaluate_model(xgb, X_test, y_test, "XGBoost", le)
    joblib.dump(xgb, os.path.join(OUTPUT_DIR, "xgboost_baseline.pkl"))

    print("="*60)
    print("[DONE] Baseline complete.")
    print("       Compare these results against stage4_train_eval.py")
    print("       (LLM and CTGAN augmented runs) on the same real_test.csv.")
    print("="*60 + "\n")

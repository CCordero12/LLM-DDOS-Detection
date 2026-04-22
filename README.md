# LLM-Based Synthetic Data Augmentation for DDoS Detection

This project investigates whether LLM-generated and CTGAN-generated synthetic network flow data can improve the robustness of machine learning-based Network Intrusion Detection Systems (NIDS) against DDoS attacks.

**Dataset:** CIC-DDoS2019  
**Task:** Multi-class DDoS attack detection (19 classes: 1 benign + 18 attack types)  
**Models evaluated:** Random Forest, XGBoost  
**Synthetic generation methods:** Mistral-7B-Instruct-v0.2 (LLM), CTGAN

---

## Key Findings

| Condition | XGBoost Accuracy | Macro F1 | Detection Rate | False Alarm Rate |
|---|---|---|---|---|
| Baseline (real data only) | 63.56% | 63.79% | 99.89% | 1.00% |
| + LLM synthetic | **65.68%** | **65.67%** | 99.83% | **0.00%** |
| + CTGAN synthetic | 63.72% | 63.65% | 99.83% | 2.00% |

LLM augmentation improved accuracy by +2.12 pp and eliminated false alarms. CTGAN matched detection rate but doubled the false alarm rate — operationally unacceptable for a production NIDS.

---

## Pipeline

```
Stage 1 & 2  →  Stage 3 (LLM)  ──┐
                Stage 3b (CTGAN) ──┤→  Stage 3c (combine)  →  Stage 4
```

| Script | Stage | Description |
|---|---|---|
| `stage1_2_nids.py` | 1 & 2 | Load and preprocess CIC-DDoS2019; train baseline RF + XGBoost |
| `stage3_llm_generation.py` | 3 | Generate synthetic attack flows using Mistral-7B |
| `stage3b_ctgan.py` | 3b | Generate synthetic flows using CTGAN |
| `combine.py` | 3c | Merge real + synthetic into training sets; create held-out test set |
| `stage4_baseline.py` | 4a | Train and evaluate on real data only (baseline) |
| `stage4_train_eval.py` | 4b | Train and evaluate on augmented data (LLM and CTGAN conditions) |

---

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost ctgan matplotlib seaborn joblib transformers torch
```

A CUDA-capable GPU is strongly recommended for Stage 3 (LLM generation). Stage 3b (CTGAN) runs on CPU but is faster on GPU.

---

## How to Run

### 1. Prepare the dataset

Download the [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) and place the CSV files in a data directory.

Edit `DATA_DIR` and `OUTPUT_DIR` in `stage1_2_nids.py` to point to your paths, then run:

```bash
python stage1_2_nids.py
```

This produces preprocessed data, trained baseline models, and saves artifacts to `./outputs/`.

> **Note:** The downstream stages (3, 3b, 3c, 4) expect a `sample.csv` in `./outputs/` — a stratified sample of 500 rows per class drawn from the full dataset. Create this from your preprocessed data before proceeding.

---

### 2. Generate synthetic flows — LLM (Mistral-7B)

```bash
python stage3_llm_generation.py
```

Prompts Mistral-7B-Instruct-v0.2 to generate synthetic network flows that blend attack and benign statistical properties. Outputs `./outputs/synthetic_flows.csv`.

Configuration (edit at top of script):
- `MODEL_NAME` — swap to any HuggingFace causal LM
- `N_ROUNDS` / `N_SYNTHETIC` — number of generation rounds and flows per round
- `MAX_FEATURES` — number of features included in the prompt (default: 20)

---

### 3. Generate synthetic flows — CTGAN

```bash
python stage3b_ctgan.py
```

Trains a Conditional Tabular GAN on real attack + benign flows and generates synthetic flows. Outputs `./outputs/synthetic_flows_ctgan.csv`.

---

### 4. Combine real and synthetic data

```bash
python combine.py
```

Splits real data 80/20 into train/test, appends synthetic flows to the training split, and saves:
- `./outputs/combined_train_llm.csv` — real + LLM synthetic
- `./outputs/combined_train_ctgan.csv` — real + CTGAN synthetic
- `./outputs/real_test.csv` — held-out real data only (shared evaluation set)

---

### 5. Run baseline evaluation

```bash
python stage4_baseline.py
```

Trains RF and XGBoost on real data only and evaluates on `real_test.csv`. Saves metrics and confusion matrices to `./outputs/`.

---

### 6. Run augmented evaluation

```bash
python stage4_train_eval.py
```

Trains RF and XGBoost on both augmented datasets (LLM and CTGAN) and evaluates both on the same `real_test.csv`. Saves per-condition metrics, confusion matrices, and a comparison summary to `./outputs/comparison_summary.csv`.

---

## Outputs

All results are written to `./outputs/`:

| File | Description |
|---|---|
| `real_test.csv` | Held-out test set (real flows only) |
| `synthetic_flows.csv` | LLM-generated synthetic flows |
| `synthetic_flows_ctgan.csv` | CTGAN-generated synthetic flows |
| `combined_train_llm.csv` | Augmented training set (LLM condition) |
| `combined_train_ctgan.csv` | Augmented training set (CTGAN condition) |
| `*_metrics.csv` | Classification metrics per model/condition |
| `*_ddos_metrics.csv` | DDoS detection metrics (DR, FAR, per-class) |
| `*_confusion_matrix.png` | Confusion matrices |
| `comparison_summary.csv` | Side-by-side DR/FAR summary across all conditions |

---

## Experiment Configuration

| Parameter | Value |
|---|---|
| Dataset | CIC-DDoS2019 |
| Sample size | 9,439 flows (500 per class) |
| Train/test split | 80/20 stratified, seed=42 |
| Test set size | 1,888 real flows |
| LLM model | mistralai/Mistral-7B-Instruct-v0.2 |
| LLM synthetic flows | 469 (labelled `ATTACK_SYNTHETIC`) |
| CTGAN synthetic flows | 1,000 (500 attack + 500 benign) |
| CTGAN epochs | 300 |
| RF estimators | 100 |
| XGBoost estimators | 100, max\_depth=6, lr=0.1 |
| Feature scaling | StandardScaler (fit on train only) |

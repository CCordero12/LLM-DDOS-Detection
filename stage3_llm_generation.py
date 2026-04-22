"""
=============================================================
NIDS - CIC-DDoS2019 Dataset
Stage 3: LLM-Based Conditional Flow Generation (Standalone)
Model: LLaMA / Mistral via HuggingFace Transformers
=============================================================

Goal:
    Sample real attack and benign flows directly from the dataset.
    Prompt the LLM to generate synthetic flows that blend both
    distributions — preserving attack semantics while reducing
    the statistical signals that ML classifiers rely on.

No dependency on Stage 1 or Stage 2.
Outputs: synthetic_flows.csv
=============================================================
"""

import os
import re
import json
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =============================================================
# CONFIGURATION
# =============================================================

# Swap models as needed:
# LLaMA 3:  "meta-llama/Meta-Llama-3-8B-Instruct"
# Mistral:  "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Fast mode: loads from sample.csv (recommended for Stage 3) ---
# --- Full mode: loads all 18 CSVs (slow, not needed for LLM generation) ---
USE_SAMPLE  = True
SAMPLE_PATH = "./outputs/sample.csv"
DATA_DIR    = "/lclhome/mjawa009/LlmDal/cic ddos 2019/archive (1)"

OUTPUT_DIR = "./outputs"

N_SHOT_ATTACK  = 3
N_SHOT_BENIGN  = 3
N_SYNTHETIC    = 10
N_ROUNDS       = 50
MAX_FEATURES   = 20
RANDOM_STATE   = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# STEP 1 — LOAD AND SAMPLE REAL FLOWS
# =============================================================

def load_and_sample(n_attack, n_benign, max_features):
    """
    Load data either from sample.csv (fast) or full CSVs (slow).
    Sample attack and benign flows for the LLM prompt.
    """

    if USE_SAMPLE:
        print(f"[INFO] Fast mode — loading from: {SAMPLE_PATH}")
        df = pd.read_csv(SAMPLE_PATH, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"[INFO] Sample records loaded: {len(df):,}")
    else:
        csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")
        print(f"[INFO] Full mode — loading {len(csv_files)} CSV file(s)...")
        dfs = []
        for i, f in enumerate(csv_files, 1):
            print(f"[INFO] Reading file {i}/{len(csv_files)}: {os.path.basename(f)}")
            dfs.append(pd.read_csv(f, low_memory=False))
        print("[INFO] Concatenating all files...")
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        print(f"[INFO] Total records: {len(df):,}")

    # Find label column
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError("Label column not found in dataset.")

    df[label_col] = df[label_col].str.strip()

    # Keep only numeric feature columns
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = feature_cols[:max_features]

    # Clean data
    df = df[selected_features + [label_col]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"[INFO] Using {len(selected_features)} features.")
    print(f"[INFO] Class distribution:\n{df[label_col].value_counts()}\n")

    # Sample attack flows
    attack_df = df[df[label_col].str.upper() != "BENIGN"].sample(
        n=n_attack, random_state=RANDOM_STATE
    )[selected_features]

    # Sample benign flows
    benign_df = df[df[label_col].str.upper() == "BENIGN"].sample(
        n=n_benign, random_state=RANDOM_STATE
    )[selected_features]

    print(f"[INFO] Sampled {n_attack} attack flows and {n_benign} benign flows.\n")
    return attack_df, benign_df, selected_features


# =============================================================
# STEP 2 — BUILD THE PROMPT
# =============================================================

def build_prompt(attack_df, benign_df, selected_features, n_synthetic):
    attack_examples = attack_df.round(4).to_dict(orient="records")
    benign_examples = benign_df.round(4).to_dict(orient="records")

    attack_json = json.dumps(attack_examples, indent=2)
    benign_json = json.dumps(benign_examples, indent=2)
    feature_list = ", ".join(selected_features)

    # [INST] tags required for Mistral to follow instructions properly
    prompt = f"""[INST] You are a network traffic generator for cybersecurity research.

Your task is to generate synthetic network flow records that blend the statistical properties of attack traffic and benign traffic. These synthetic flows should preserve the underlying attack behavior while subtly shifting feature values toward benign ranges to reduce detectability by machine learning classifiers.

Here are real ATTACK flow examples:
{attack_json}

Here are real BENIGN flow examples:
{benign_json}

Generate exactly {n_synthetic} synthetic flow records. Each record must:
- Contain exactly these fields: {feature_list}
- Use numeric values only (no strings, no nulls, no NaN)
- Blend statistical patterns from both attack and benign examples
- Shift feature values subtly toward benign ranges while still representing attack-like behavior

CRITICAL: Return ONLY a valid JSON array. No explanation, no markdown, no code blocks, no extra text. Start your response directly with [ and end with ]. [/INST]"""

    return prompt


# =============================================================
# STEP 3 — LOAD THE LLM
# =============================================================

def load_llm(model_name):
    print(f"[INFO] Loading model: {model_name}")
    print("[INFO] This may take a few minutes on first run...\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,    # increased from 2048 to avoid truncation
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Model loaded on: {device}\n")
    return pipe


# =============================================================
# STEP 4 — GENERATE AND PARSE FLOWS
# =============================================================

def clean_llm_output(text):
    """Strip markdown fences and extract the JSON array."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "").strip()
    try:
        start = text.index("[")
        end   = text.rindex("]") + 1
        return text[start:end]
    except ValueError:
        return text


def generate_flows(pipe, prompt):
    print("[INFO] Generating...")
    output = pipe(prompt)[0]["generated_text"]
    cleaned = clean_llm_output(output)
    try:
        flows = json.loads(cleaned)
        print(f"[INFO] Parsed {len(flows)} flows successfully.")
        return flows
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse output: {e}")
        print(f"[DEBUG] Raw snippet:\n{output[:400]}\n")
        return []


# =============================================================
# STEP 5 — VALIDATE FLOWS
# =============================================================

def validate_flows(flows, selected_features):
    valid = []
    for i, flow in enumerate(flows):
        try:
            record = {f: float(flow[f]) for f in selected_features}
            valid.append(record)
        except (KeyError, TypeError, ValueError) as e:
            print(f"[WARNING] Dropping flow {i}: {e}")

    print(f"[INFO] {len(valid)}/{len(flows)} flows passed validation.\n")
    return pd.DataFrame(valid, columns=selected_features)


# =============================================================
# MAIN PIPELINE
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  STAGE 3 — LLM CONDITIONAL FLOW GENERATION")
    mode = "FAST (sample.csv)" if USE_SAMPLE else "FULL (all CSVs)"
    print(f"  Mode: {mode}")
    print("="*55 + "\n")

    # Step 1 — Load and sample
    attack_df, benign_df, selected_features = load_and_sample(
        N_SHOT_ATTACK, N_SHOT_BENIGN, MAX_FEATURES
    )

    # Step 2 — Build prompt
    prompt = build_prompt(attack_df, benign_df, selected_features, N_SYNTHETIC)
    print("[INFO] Prompt preview (first 400 chars):")
    print(prompt[:400])
    print("...\n")

    # Step 3 — Load LLM
    pipe = load_llm(MODEL_NAME)

    # Step 4 & 5 — Generate and validate over N rounds
    all_synthetic = []
    for round_num in range(1, N_ROUNDS + 1):
        print(f"[INFO] Round {round_num}/{N_ROUNDS}...")
        flows = generate_flows(pipe, prompt)
        if flows:
            validated = validate_flows(flows, selected_features)
            if not validated.empty:
                all_synthetic.append(validated)

    # Combine and save
    if all_synthetic:
        synthetic_df = pd.concat(all_synthetic, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "synthetic_flows.csv")
        synthetic_df.to_csv(out_path, index=False)
        print(f"[DONE] {len(synthetic_df)} synthetic flows saved to: {out_path}")
        print(f"       Features: {list(synthetic_df.columns)}")
    else:
        print("[ERROR] No valid synthetic flows were generated.")

    print("\n[DONE] Stage 3 complete. Ready for Stage 4 — Adversarial Evaluation.\n")
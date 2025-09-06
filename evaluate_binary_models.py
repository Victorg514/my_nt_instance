
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
import pathlib
import warnings

from transformers import AutoTokenizer, RobertaModel, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight # Needed for feature selection context

warnings.filterwarnings('ignore')

# --- This MUST be the same AdvancedDeltaModel class from your binary training script ---
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()); self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid()); self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size); self.classifier_dropout = nn.Dropout(dropout_rate); self.classifier = nn.Linear(hidden_size, num_labels); self.feature_importance = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state; attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None); cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1); text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled; behavioral_features = self.feature_encoder(delta.float()); combined_features = torch.cat([text_features, behavioral_features], dim=1); gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values; pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output); logits = self.classifier(dropped_output)
        return {"logits": logits}


def get_feature_list(df, n_features, binary_y):
    """Gets the deterministic list of top N features for a binary task."""
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    if not feature_cols: return []
    X = df[feature_cols].fillna(0).values; y = binary_y
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    top_features = feature_scores.head(n_features)['feature'].tolist()
    return top_features

def load_all_binary(positive_class_name: str):
    """
    Main function to load/run predictions for a specific binary task.
    """
    # --- DYNAMIC CONFIGURATION based on the task ---
    DATA_FILE = pathlib.Path("data/deviation_advanced_final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    BASELINE_DIR = f"out/baseline_binary_{positive_class_name}"
    DELTA_DIR = f"out/delta_binary_{positive_class_name}"
    MAX_FEATURES_TO_SELECT = 50

    # Dynamic cache paths to avoid overwriting
    CACHE = {
        "proba_plain": f"cache/proba_plain_binary_{positive_class_name}.npy",
        "proba_delta": f"cache/proba_delta_binary_{positive_class_name}.npy",
        "pred_plain":  f"cache/pred_plain_binary_{positive_class_name}.npy",
        "pred_delta":  f"cache/pred_delta_binary_{positive_class_name}.npy",
        "labels":      f"cache/labels_binary_{positive_class_name}.npy"
    }
    
    os.makedirs("cache", exist_ok=True)
    if all(os.path.exists(CACHE[f]) for f in CACHE):
        print(f"--- Loading BINARY predictions for '{positive_class_name}' from cache ---")
        return {
            "proba_plain": np.load(CACHE["proba_plain"]),
            "proba_delta": np.load(CACHE["proba_delta"]),
            "pred_plain":  np.load(CACHE["pred_plain"]),
            "pred_delta":  np.load(CACHE["pred_delta"]),
            "labels":      np.load(CACHE["labels"])
        }

    print(f"--- Running BINARY Inference for '{positive_class_name}' (Cache not found) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load and Relabel Data ---
    df = pd.read_parquet(DATA_FILE)
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if positive_class_name == 'mental_health':
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else:
        positive_label_id = LABEL_MAP[positive_class_name]
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)

    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Baseline Prediction ---
    print("\nEvaluating Baseline Binary Model...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_DIR).to(device)
    baseline_model.eval()
    
    def tok_only(batch): return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    plain_ds = Dataset.from_pandas(test_df).map(tok_only, batched=True, remove_columns=test_df.columns.tolist())
    plain_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    from torch.utils.data import DataLoader
    plain_loader = DataLoader(plain_ds, batch_size=64)
    all_baseline_logits = []
    with torch.no_grad():
        for batch in tqdm(plain_loader, desc=f"Predicting Baseline ({positive_class_name})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = baseline_model(**batch).logits
            all_baseline_logits.append(logits.cpu().numpy())
    proba_plain = np.concatenate(all_baseline_logits)

    # --- Delta Prediction ---
    print("\nEvaluating Delta Binary Model...")
    train_df = df[df['split'] == 'train']
    selected_features = get_feature_list(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    actual_num_features = len(selected_features)
    
    delta_model = AdvancedDeltaModel(base_name=MODEL_NAME, num_labels=2, num_features=actual_num_features).to(device)
    delta_model.load_state_dict(torch.load(f"{DELTA_DIR}/pytorch_model.bin", map_location=device))
    delta_model.eval()

    def tok_plus(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        return encoding

    delta_ds = Dataset.from_pandas(test_df).map(tok_plus, batched=True, remove_columns=test_df.columns.tolist())
    delta_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
    
    delta_loader = DataLoader(delta_ds, batch_size=32)
    all_delta_logits = []
    with torch.no_grad():
        for batch in tqdm(delta_loader, desc=f"Predicting Delta ({positive_class_name})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = delta_model(**batch)['logits']
            all_delta_logits.append(logits.cpu().numpy())
    proba_delta = np.concatenate(all_delta_logits)
    
    # --- Process and Cache All Results ---
    labels = test_df['binary_label'].values
    pred_plain = proba_plain.argmax(axis=1)
    pred_delta = proba_delta.argmax(axis=1)

    print("\nCaching binary predictions for future runs...")
    np.save(CACHE["proba_plain"], proba_plain)
    np.save(CACHE["proba_delta"], proba_delta)
    np.save(CACHE["labels"], labels)
    np.save(CACHE["pred_plain"], pred_plain)
    np.save(CACHE["pred_delta"], pred_delta)

    return {
        "proba_plain": proba_plain,
        "proba_delta": proba_delta,
        "pred_plain":  pred_plain,
        "pred_delta":  pred_delta,
        "labels":      labels
    }


def main(args):
    """When run as a script, this function produces the detailed text reports."""
    
    positive_class_name = args.positive_class
    
    # Run predictions for the specified binary task
    d = load_all_binary(positive_class_name)
    
    # Unpack the dictionary
    labels = d["labels"]
    pred_plain = d["pred_plain"]
    pred_delta = d["pred_delta"]
    
    if positive_class_name == 'mental_health':
        positive_class_label = "mental_health"
    else:
        positive_class_label = positive_class_name
    
    target_names = ["other", positive_class_label]
    
    # --- Generate Reports ---
    print("\n" + "="*70)
    print(f"--- BASELINE MODEL RESULTS ({positive_class_label.upper()} vs. OTHERS) ---")
    print("="*70)
    print(classification_report(labels, pred_plain, target_names=target_names, digits=4))
    
    print("\n" + "="*70)
    print(f"--- PERSONALIZED (DELTA) MODEL RESULTS ({positive_class_label.upper()} vs. OTHERS) ---")
    print("="*70)
    print(classification_report(labels, pred_delta, target_names=target_names, digits=4))
    
    print("\n" + "="*70)
    print("--- CONFUSION MATRICES ---")
    print("="*70)
    print("\nBaseline Model:")
    print(pd.DataFrame(confusion_matrix(labels, pred_plain), index=target_names, columns=target_names))
    
    print("\nPersonalized (Delta) Model:")
    print(pd.DataFrame(confusion_matrix(labels, pred_delta), index=target_names, columns=target_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive_class", type=str, required=True,
        choices=['mental_health', 'depression', 'anxiety', 'bipolar'],
        help="The binary classification task to evaluate."
    )
    args = parser.parse_args()
    main(args)
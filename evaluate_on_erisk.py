# evaluate_on_erisk.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RobertaModel
from datasets import Dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# You MUST paste the exact same AdvancedDeltaModel class definition here
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()); self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid()); self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size); self.classifier_dropout = nn.Dropout(dropout_rate); self.classifier = nn.Linear(hidden_size, num_labels); self.feature_importance = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state; attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None); cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1); text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled; behavioral_features = self.feature_encoder(delta.float()); combined_features = torch.cat([text_features, behavioral_features], dim=1); gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values; pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output); logits = self.classifier(dropped_output)
        return {"logits": logits}

def main():
    print("--- Evaluating Twitter-Trained Screener Model on eRisk Reddit Data ---")
    DATA_FILE = "data/erisk_processed_for_testing.parquet"
    TOKENIZER_NAME = "mental/mental-roberta-base"
    # This is the 'Control vs Others' binary model you trained
    MODEL_PATH = "out/delta_binary_mental_health" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_df = pd.read_parquet(DATA_FILE)
    
    # Dynamically find feature columns from the processed file
    exclude_cols = ['user_id', 'label', 'label_name', 'text']
    selected_features = [c for c in test_df.columns if c not in exclude_cols]
    num_features = len(selected_features)
    print(f"Using {num_features} features from the processed eRisk file.")

    model = AdvancedDeltaModel(base_name=TOKENIZER_NAME, num_labels=2, num_features=num_features).to(device)
    model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    def tokenize_with_features(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        return encoding

    test_ds = Dataset.from_pandas(test_df).map(tokenize_with_features, batched=True, remove_columns=test_df.column_names)
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32)
    
    final_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on eRisk"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)['logits']
            final_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            
    final_preds = np.concatenate(final_preds)
    true_labels = test_df['label'].values
    
    print("\n" + "="*70)
    print("--- FINAL CROSS-PLATFORM PERFORMANCE REPORT (eRisk) ---")
    print("="*70)
    print(classification_report(
        true_labels, final_preds, target_names=["control", "depression"], digits=4
    ))

if __name__ == "__main__":
    main()
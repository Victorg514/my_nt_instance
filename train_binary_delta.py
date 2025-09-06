
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    RobertaModel,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
import pathlib
import warnings
warnings.filterwarnings('ignore')

# (AdvancedDeltaModel and WeightedDeltaTrainer classes are unchanged)
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()); self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid()); self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size); self.classifier_dropout = nn.Dropout(dropout_rate); self.classifier = nn.Linear(hidden_size, num_labels); self.feature_importance = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state; attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None); cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1); text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled; behavioral_features = self.feature_encoder(delta.float()); combined_features = torch.cat([text_features, behavioral_features], dim=1); gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values; pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output); logits = self.classifier(dropped_output)
        return logits
class WeightedDeltaTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs); self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        logits = model(**inputs) # Model now returns logits directly
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        # For evaluation, we need to return logits as well
        return (loss, {"logits": logits}) if return_outputs else loss

# --- THIS IS THE CORRECTED FEATURE SELECTION FUNCTION ---
def select_best_features(df, n_features, binary_y):
    """
    Selects features, prioritizing the powerful deviation and z-scored columns.
    """
    # Define a list of high-quality, pre-scaled feature names
    core_features = [
        'len_z', 'sent_compound_z', 'circadian_deviation', 'anomaly_score',
        'tpd_3d_dev', 'tpd_7d_dev', 'tpd_14d_dev', 'tpd_30d_dev',
        'emotion_volatility_10tw', 'polarity_shift', 'self_focus',
        'isolation_score', 'manic_indicator', 'depressive_indicator'
    ]
    
    # Find which of these core features actually exist in the dataframe
    available_core_features = [f for f in core_features if f in df.columns]
    
    # If we need more features, add other numeric columns
    other_features = [
        c for c in df.columns if 
        df[c].dtype in ['float64', 'int64'] and 
        c not in available_core_features and 
        c not in ['tweet_id', 'user_id', 'label', 'label_name', 'split', 'created', 'binary_label']
    ]
    
    # Combine the lists, with core features first
    all_possible_features = available_core_features + other_features
    
    # Return up to the requested number of features
    selected = all_possible_features[:n_features]
    
    print(f"Found {len(all_possible_features)} total numerical features.")
    print(f"Selected {len(selected)} best features for the binary task.")
    return selected
# --------------------------------------------------------

def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"--- Training Binary Classifier for: {POSITIVE_CLASS_NAME.upper()} vs. OTHERS ---")

    # (Configuration and data prep are unchanged)
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/delta_binary_{POSITIVE_CLASS_NAME}" # New output dir
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42
    MAX_FEATURES_TO_SELECT = 50
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA_FILE)
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health': df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else: positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]; df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)
    train_df = df[df['split'] == 'train']
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['binary_label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Binary class weights: {class_weights}")
    
    # Call the new, robust feature selector
    selected_features = select_best_features(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    actual_num_features = len(selected_features)
    if actual_num_features == 0:
        raise ValueError("No valid features selected for training.")

    # (The rest of the script is the same)
    def create_dataset(split): return Dataset.from_pandas(df[df['split'] == split].reset_index(drop=True))
    train_ds, val_ds = create_dataset("train"), create_dataset("val")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_with_features(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        encoding["labels"] = batch["binary_label"]
        return encoding
    train_ds = train_ds.map(tokenize_with_features, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_with_features, batched=True, remove_columns=val_ds.column_names)
    model = AdvancedDeltaModel(base_name=MODEL_NAME, num_labels=2, num_features=actual_num_features).to(device)


    def compute_metrics(eval_pred):
        # The Trainer passes the dictionary from compute_loss here.
        # So, eval_pred.predictions is {'logits': logits_array}
        logits = eval_pred.predictions['logits']
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(labels, preds, average='binary')
        return {"f1": f1}
    
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        
        
        learning_rate=5e-5,          # Set to the more aggressive learning rate
        warmup_ratio=0.1,            # Added for training stability
        weight_decay=0.05,           # Added for regularization

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )
    trainer = WeightedDeltaTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"--- Finished training and saved model to {OUT_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression', 'anxiety', 'bipolar'], help="The class to treat as the positive label (1).")
    args = parser.parse_args()
    main(args)
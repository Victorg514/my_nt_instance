

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


class AdvancedDeltaModel(nn.Module):
    """
    Full class definition for the delta model.
    --- MODIFIED: The forward pass now ONLY returns logits. ---
    """
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.base = RobertaModel.from_pretrained(base_name)
        hidden_size = self.base.config.hidden_size
        self.feature_encoder = nn.Sequential(nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid())
        self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size)
        self.classifier_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.feature_importance = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state
        attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1)
        text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled
        behavioral_features = self.feature_encoder(delta.float())
        combined_features = torch.cat([text_features, behavioral_features], dim=1)
        gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values
        pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output)
        logits = self.classifier(dropped_output)
        
        # The Trainer expects a tuple if labels are passed, so we return loss=None
        loss = None
        if labels is not None:
            # We don't calculate loss here anymore, but must return a tuple
            return (None, logits)
        return logits




class WeightedDeltaTrainer(Trainer):
    """
    Custom trainer that explicitly applies the weighted loss.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # -----------------------
        labels = inputs.pop("labels")
        # Get logits from the model
        outputs = model(**inputs)
        logits = outputs[1] if isinstance(outputs, tuple) else outputs
        
        # Explicitly apply the weighted loss
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        
        return (loss, outputs) if return_outputs else loss


def select_best_features(df, n_features, binary_y):
    # (This function is unchanged)
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    if not feature_cols: return []
    X = df[feature_cols].fillna(0).values; y = binary_y
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    top_features = feature_scores.head(n_features)['feature'].tolist()
    print(f"Selected {len(top_features)} best features for the binary task.")
    return top_features


def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"--- Training Binary Classifier for: {POSITIVE_CLASS_NAME.upper()} vs. OTHERS ---")

    # (Configuration and data prep are unchanged)
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/delta_binary_{POSITIVE_CLASS_NAME}"
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 2, 42
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
    selected_features = select_best_features(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    actual_num_features = len(selected_features)
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
    
    # --- THIS IS THE CORRECTED compute_metrics FUNCTION ---
    def compute_metrics(eval_pred):
        # The output is a tuple (loss, logits), so logits are at index 1
        logits = eval_pred.predictions[1]
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(labels, preds, average='binary')
        return {"f1": f1}
    # --------------------------------------------------------

    # (TrainingArguments are unchanged)
    training_args = TrainingArguments(
        output_dir=OUT_DIR, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        fp16=False, report_to="none", seed=SEED,
        remove_unused_columns=False,
    )
    
    # --- USE THE NEW WeightedDeltaTrainer ---
    trainer = WeightedDeltaTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor # Pass the weights to the trainer
    )
    # ----------------------------------------
    
    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"--- Finished training and saved model to {OUT_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive_class", type=str, required=True,
        choices=['mental_health', 'depression', 'anxiety', 'bipolar'],
        help="The class to treat as the positive label (1)."
    )
    args = parser.parse_args()
    main(args)
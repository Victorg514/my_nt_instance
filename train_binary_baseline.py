# train_binary_baseline.py

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import pathlib
import warnings
warnings.filterwarnings('ignore')

class WeightedLossTrainer(Trainer):
    """
    Custom trainer to handle class weights correctly.
    Added **kwargs to compute_loss for future compatibility.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss

def main(args):
    # --- Configuration from args ---
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- Training BINARY BASELINE Classifier for: {POSITIVE_CLASS_NAME.upper()} vs. OTHERS ---")

    # --- Static Configuration ---
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/baseline_binary_{POSITIVE_CLASS_NAME}"
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load and Relabel Data ---
    df = pd.read_parquet(DATA_FILE)
    df = df[["text", "label", "label_name", "split"]] # Baseline uses text only

    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health':
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else:
        positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)

    # Re-compute class weights for the binary problem
    train_df = df[df['split'] == 'train']
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['binary_label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Binary class weights computed: {class_weights}")

    # --- Create Datasets ---
    def create_dataset(split): return Dataset.from_pandas(df[df['split'] == split].reset_index(drop=True))
    train_ds, val_ds = create_dataset("train"), create_dataset("val")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        encoding["labels"] = batch["binary_label"]
        return encoding
    
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    # --- Model and Trainer ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        f1 = f1_score(eval_pred.label_ids, preds, average='binary')
        return {"f1": f1}

    training_args = TrainingArguments(
        output_dir=OUT_DIR, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        fp16=False, report_to="none", seed=SEED,
        label_smoothing_factor=0.1,
    )

    trainer = WeightedLossTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor
    )
    
    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"--- Finished training and saved BASELINE model to {OUT_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive_class", type=str, required=True,
        choices=['mental_health', 'depression', 'anxiety', 'bipolar'],
        help="The class to treat as the positive label (1)."
    )
    args = parser.parse_args()
    main(args)
"""Fine-tune DeBERTa-v3-base as a binary poison classifier (Stage 1).

Usage:
    python scripts/train_stage1.py \
        --train-data data/synthetic_train/train.jsonl \
        --val-data data/synthetic_train/val.jsonl \
        --output-dir results/stage1_classifier
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, fbeta_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    STAGE1_BATCH_SIZE,
    STAGE1_LEARNING_RATE,
    STAGE1_MAX_LENGTH,
    STAGE1_MODEL,
    STAGE1_MODEL_DIR,
    STAGE1_TRAIN_EPOCHS,
)
from src.attacks.generate_train_data import load_split


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["query"],
        batch["document"],
        truncation=True,
        max_length=STAGE1_MAX_LENGTH,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f2 = fbeta_score(labels, preds, beta=2, average="binary")
    return {"f2": f2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", default=STAGE1_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=STAGE1_TRAIN_EPOCHS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STAGE1_MODEL, num_labels=2
    )

    train_raw = load_split("train", data_dir=os.path.dirname(args.train_data))
    val_raw = load_split("val", data_dir=os.path.dirname(args.val_data))

    train_ds = Dataset.from_list(train_raw).map(
        lambda b: tokenize(b, tokenizer), batched=True
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda b: tokenize(b, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        per_device_eval_batch_size=STAGE1_BATCH_SIZE,
        learning_rate=STAGE1_LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f2",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved best model to {args.output_dir}")

    # Evaluate on test set
    test_raw = load_split("test", data_dir=os.path.dirname(args.train_data))
    test_ds = Dataset.from_list(test_raw).map(
        lambda b: tokenize(b, tokenizer), batched=True
    )
    results = trainer.evaluate(test_ds)
    print("Test F2:", results.get("eval_f2"))


if __name__ == "__main__":
    main()

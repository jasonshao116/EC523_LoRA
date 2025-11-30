import argparse, os, json, inspect
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2ForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

@dataclass
class Args:
    model_name: str = "gpt2"
    max_steps: int = 800
    lr: float = 5e-5          # usually smaller for full FT than LoRA
    batch_size: int = 16
    eval_batch_size: int = 32
    output_dir: str = "runs/gpt2_sst2_full"
    max_length: int = 256

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--output_dir", type=str, default="runs/gpt2_sst2_full")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()

def build_training_args(**kwargs) -> TrainingArguments:
    # Make transformer-version-agnostic construction
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in sig.parameters:
        allowed["do_eval"] = True
    return TrainingArguments(**allowed)

def main():
    args = parse_args()

    # Data: SST-2
    ds = load_dataset("glue", "sst2")
    text_key, label_key = "sentence", "label"

    # Tokenizer (GPT-2 has no pad token -> use eos as pad)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def tokenize(batch):
        enc = tok(batch[text_key], truncation=True, max_length=args.max_length)
        enc["labels"] = batch[label_key]
        return enc

    keep_cols = [label_key]
    remove_cols = [c for c in ds["train"].column_names if c not in keep_cols]
    ds = ds.map(tokenize, batched=True, remove_columns=remove_cols)

    collator = DataCollatorWithPadding(tok)

    # Model (no LoRA, full fine-tuning)
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.problem_type = "single_label_classification"

    # Make sure all params are trainable
    for _, p in model.named_parameters():
        p.requires_grad = True

    # Print trainable vs total params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Trainer
    training_args = build_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        save_total_limit=1,
        max_steps=args.max_steps,
        report_to=[],
        remove_unused_columns=True,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    # Save predictions for eval
    preds = trainer.predict(ds["validation"]).predictions.argmax(-1)

    # Convert to plain Python lists (avoid HuggingFace Column / NumPy types)
    preds_list = [int(x) for x in preds.tolist()]
    labels_list = list(ds["validation"][label_key])  # <- force to standard list

    os.makedirs("runs", exist_ok=True)
    with open("runs/sst2_val_preds_full.json", "w") as f:
        json.dump(preds_list, f)
    with open("runs/sst2_val_labels_full.json", "w") as f:
        json.dump(labels_list, f)

if __name__ == "__main__":
    main()

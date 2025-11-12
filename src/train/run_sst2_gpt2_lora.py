import argparse, os, json, inspect
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2ForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.lora.inject import inject_lora_gpt2

@dataclass
class Args:
    model_name: str = "gpt2"
    max_steps: int = 800
    lr: float = 2e-4
    batch_size: int = 16
    eval_batch_size: int = 32
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    output_dir: str = "runs/gpt2_sst2_lora"
    max_length: int = 256

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--output_dir", type=str, default="runs/gpt2_sst2_lora")
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

    # Model
    model = GPT2ForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    model.config.problem_type = "single_label_classification"

    # Inject LoRA into GPT-2 blocks (attn + mlp)
    model = inject_lora_gpt2(model, r=args.r, alpha=args.alpha, dropout=args.dropout, verbose=True)

    # Freeze base params; train only LoRA A/B and classifier head (score)
    for n, p in model.named_parameters():
        train = ("A" in n) or ("B" in n) or n.startswith("score.")
        p.requires_grad = train

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

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
    with open("runs/sst2_val_preds.json", "w") as f:
        json.dump(preds_list, f)
    with open("runs/sst2_val_labels.json", "w") as f:
        json.dump(labels_list, f)


if __name__ == "__main__":
    main()

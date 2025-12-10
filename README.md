# LoRA Fine-tuning 

This repo contains our scripts to perform LoRA fine-tuning on GPT-2 Small and SST-2, as well as other fine-tuning methods (full fine-tuning, adapter tuning, and prompt tuning) on the same task. 

## Approach  

### src/lora/layers.py
Defines the tiny LoRALinear module. It keeps the original weight W frozen and adds a low-rank update B @ A * (alpha/r). Exposes:
- forward(x): computes base linear + LoRA update.
- merge_weights_(): (optional) bakes the LoRA update into W for inference.

### src/lora/inject.py
Walks through the GPT-2 model and swaps specific linear-like layers for LoRALinear:
- Targets: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj in every block.
- Handles GPT-2’s Conv1D (weight layout differs from nn.Linear).
- Prints which layers were wrapped so we can verify LoRA placement. 

### src/train/run_sst2_gpt2_lora.py 
The main training runner for SST-2 with GPT-2 + LoRA:
- Loads SST-2 via Datasets, tokenizes, sets GPT-2 pad_token=eos_token.
- Builds GPT2ForSequenceClassification, injects LoRA, freezes base weights, leaves LoRA A/B + classifier trainable.
- Trains with Trainer, then saves validation predictions and labels to runs/sst2_val_*.json.
- Prints trainable-param count and a layer-wrap summary (sanity checks).

### src/train/eval_cls.py 
A tiny evaluator:
- Loads runs/sst2_val_preds.json and runs/sst2_val_labels.json.
- Reports Accuracy and F1 (sklearn). Useful for quick comparisons after each run. 

### tests/test_lora.py 
Minimal unit tests for LoRA:
- Checks the shape correctness of LoRALinear output.
- Calls merge_weights_() to ensure it doesn’t error. (Good smoke tests for refactors.) 

### Jupyter/ 
The directory that includes the Jupyter Notebooks for full fine-tuning, adapter tuning, and prompt tuning. 

## How to run: 

``` bash
# Set up environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# LoRA on GPT-2 small (SST-2)
python -m src.train.run_sst2_gpt2_lora --max_steps 800 --batch_size 16 --r 8 --alpha 16 --dropout 0.05
# Evaluate on validation
python -m src.train.eval_cls --preds runs/sst2_val_preds.json --labels runs/sst2_val_labels.json
``` 
Arguments in the command: 

**max_steps 800**
Run 800 optimization steps total, regardless of how many epochs that corresponds to. (800 steps ≈ 0.19 epochs) 

**batch_size 16**
Set the training batch size: each training step processes 16 examples. 

**r 8**
LoRA rank = 8

**alpha 16**
LoRA scaling factor 

**dropout 0.05**
Dropout applied before LoRA to help prevent overfitting

Above are the default values of each LoRA hyperparameter. Feel free to change them when you run.  

To run the Jupyter Notebooks of other fine-tuning methods, you can download the IPYNB files in Jupyter/ and run them in Google Colab. 

How to run: 

``` bash
# Set up environment and intstall dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# LoRA on GPT-2 small (SST-2)
python -m src.train.run_sst2_gpt2_lora --max_steps 800 --batch_size 16 --r 8 --alpha 16 --dropout 0.05
# Evaluate on validation
python -m src.train.eval_cls --preds runs/sst2_val_preds.json --labels runs/sst2_val_labels.json
``` 
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
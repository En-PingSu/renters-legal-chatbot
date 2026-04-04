"""
Unsloth LoRA finetuning — MA Tenant Law Chatbot
Model: Qwen3 4B Instruct (local)
Team 27: En-Ping Su & Maxwell Berry, CS6180

Usage:
    python Fine-Tuneing/FineTune.py
"""

import shutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from pathlib import Path
import torch

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
MODEL_PATH    = str(PROJECT_ROOT / "Fine-Tuneing" / "Qwen3")
TRAIN_DATA    = PROJECT_ROOT / "Fine-Tuneing" / "train.txt"
VAL_DATA      = PROJECT_ROOT / "Fine-Tuneing" / "val.txt"
OUTPUT_DIR    = str(PROJECT_ROOT / "Fine-Tuneing" / "lora-Qwen3")
MERGED_DIR    = str(PROJECT_ROOT / "Fine-Tuneing" / "finetuned-Qwen3")

MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
BATCH_SIZE    = 2
GRAD_ACC      = 4
EPOCHS        = 4     # epoch 3 sweet spot; load_best_model_at_end saves best val checkpoint
LR            = 2e-4

# ── Clean output dirs ─────────────────────────────────────────────────────────
for d in [OUTPUT_DIR, MERGED_DIR]:
    if Path(d).exists():
        print(f"Cleaning {d} ...")
        shutil.rmtree(d)
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_PATH,
    max_seq_length= MAX_SEQ_LEN,
    load_in_4bit  = False,
    dtype         = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = 32,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
)

print("Model loaded. Parameters:")
model.print_trainable_parameters()

# ── Load training data ────────────────────────────────────────────────────────
print(f"\nLoading training data from {TRAIN_DATA} ...")
with open(TRAIN_DATA, "r", encoding="utf-8") as f:
    raw_train = f.read()

train_samples = [s.strip() for s in raw_train.split("\n\n") if s.strip()]
print(f"Loaded {len(train_samples)} training samples")

print(f"Loading validation data from {VAL_DATA} ...")
with open(VAL_DATA, "r", encoding="utf-8") as f:
    raw_val = f.read()

val_samples = [s.strip() for s in raw_val.split("\n\n") if s.strip()]
print(f"Loaded {len(val_samples)} validation samples")

# Verify Qwen3 ChatML format
first = train_samples[0] if train_samples else ""
if "<|im_start|>" not in first:
    raise ValueError(
        "Training data is not in Qwen3 ChatML format!\n"
        "Expected <|im_start|> tokens. Run prepare_finetune_data.py first."
    )
print("Format check passed: Qwen3 ChatML tokens found.")

train_dataset = Dataset.from_dict({"text": train_samples})
val_dataset   = Dataset.from_dict({"text": val_samples})

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training ...")
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    eval_dataset       = val_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    callbacks          = [EarlyStoppingCallback(early_stopping_patience=2)],
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACC,
        warmup_steps                = 10,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        # Evaluate at end of each epoch, save best checkpoint by val loss
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 2,      # keep only 2 checkpoints on disk
        report_to                   = "none",
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete!")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# ── Save LoRA adapter (best checkpoint weights) ───────────────────────────────
print(f"\nSaving LoRA adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Merge LoRA into base weights and save ─────────────────────────────────────
print(f"\nMerging LoRA into base weights ...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR} ...")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

# Copy any missing config files from base model
print("Copying config files from base model ...")
for f in Path(MODEL_PATH).glob("*.json"):
    dest = Path(MERGED_DIR) / f.name
    if not dest.exists():
        shutil.copy(f, dest)
        print(f"  Copied {f.name}")

for f in Path(MODEL_PATH).glob("*.jinja"):
    dest = Path(MERGED_DIR) / f.name
    if not dest.exists():
        shutil.copy(f, dest)
        print(f"  Copied {f.name}")

print(f"\nDone! Merged model saved to {MERGED_DIR}")
print(f"\nNow convert to GGUF:")
print(f"  python llama.cpp\\convert_hf_to_gguf.py {MERGED_DIR} --outfile Fine-Tuneing\\finetuned-qwen3-f16.gguf --outtype f16")
print(f"\nThen serve:")
print(f"  llama.cpp\\build\\bin\\Release\\llama-server.exe -m Fine-Tuneing\\finetuned-qwen3-f16.gguf -ngl 999 --port 8080 --ctx-size 8192 --repeat-penalty 1.3 --temp 0.7")

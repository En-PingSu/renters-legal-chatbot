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
from transformers import TrainingArguments
from datasets import Dataset
from pathlib import Path
import torch

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
MODEL_PATH    = str(PROJECT_ROOT / "Fine-Tuneing" / "Qwen3")
TRAIN_DATA    = PROJECT_ROOT / "Fine-Tuneing" / "train.txt"
OUTPUT_DIR    = str(PROJECT_ROOT / "Fine-Tuneing" / "lora-Qwen3")
MERGED_DIR    = str(PROJECT_ROOT / "Fine-Tuneing" / "finetuned-Qwen3")

MAX_SEQ_LEN   = 2048
LORA_RANK     = 8
BATCH_SIZE    = 2
GRAD_ACC      = 4
EPOCHS        = 3
LR            = 1e-4

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
    lora_alpha                 = 16,   # alpha=2x rank is standard
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
)

print("Model loaded. Parameters:")
model.print_trainable_parameters()

# ── Load training data ────────────────────────────────────────────────────────
print(f"\nLoading training data from {TRAIN_DATA} ...")
with open(TRAIN_DATA, "r", encoding="utf-8") as f:
    raw = f.read()

# Split on sentinel written by prepare_finetune_data.py.
# Cannot use \n\n — RAG samples contain paragraph breaks inside the system message.
SENTINEL = "\n<<<BOUNDARY>>>\n"
samples = [s.strip() for s in raw.split(SENTINEL) if s.strip()]
print(f"Loaded {len(samples)} training samples")

# Verify Qwen3 ChatML format
first = samples[0] if samples else ""
if "<|im_start|>" not in first:
    raise ValueError(
        "Training data is not in Qwen3 ChatML format!\n"
        "Expected <|im_start|> tokens. Run prepare_finetune_data.py first."
    )
print("Format check passed: Qwen3 ChatML tokens found.")

dataset = Dataset.from_dict({"text": samples})

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training ...")
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
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
        save_strategy               = "no",
        report_to                   = "none",
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete!")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# ── Save LoRA adapter ─────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Merge LoRA into base weights and save ─────────────────────────────────────
print(f"\nMerging LoRA into base weights ...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR} ...")
model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="99GB")
tokenizer.save_pretrained(MERGED_DIR)

# Copy only non-weight config files from base model
print("Copying config files from base model ...")
SKIP_FILES = {"model.safetensors.index.json", "model.safetensors"}
for f in Path(MODEL_PATH).glob("*.json"):
    if f.name in SKIP_FILES:
        continue
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
print(f"\nServe with:")
print(f"  python Fine-Tuneing/local_server.py --model finetuned --port 8080")

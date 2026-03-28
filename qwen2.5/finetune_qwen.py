# FOR 2019 ONLY - OPTIMISED FOR G4 GPU - 96GB - 7B

# Train
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

torch.cuda.empty_cache()

# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"PyTorch version:           {torch.__version__}")
print(f"CUDA version:              {torch.version.cuda}")
print(f"GPU:                       {torch.cuda.get_device_name(0)}")
print(f"VRAM:                      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"bf16 supported:            {torch.cuda.is_bf16_supported()}")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/content/drive/MyDrive/MLP/outputs"

MAX_LENGTH        = 800
RESPONSE_TEMPLATE = "### Response:\n"

# ── Format ────────────────────────────────────────────────────────────────────
def format_example(example):
    return (
        "### Task:\n"
        "Generate live League of Legends shoutcaster commentary from the event log.\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        f"{RESPONSE_TEMPLATE}"
        f"{example['target']}"
    )

# ── Collator ──────────────────────────────────────────────────────────────────
class CompletionOnlyCollator:
    def __init__(self, response_template, tokenizer, max_length):
        self.template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        template_len = len(self.template_ids)

        for label in labels:
            for j in range(len(label) - template_len):
                if label[j:j + template_len].tolist() == self.template_ids:
                    label[:j + template_len] = -100
                    break
            else:
                label[:] = -100

        batch["labels"] = labels
        return batch

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Dataset ───────────────────────────────────────────────────────────────────
raw_train = load_dataset("json", data_files={"train": "/content/drive/MyDrive/MLP/train.jsonl"})["train"]
raw_eval  = load_dataset("json", data_files={"train": "/content/drive/MyDrive/MLP/val.jsonl"})["train"]

train_dataset = raw_train.map(lambda ex: {"text": format_example(ex)})
eval_dataset  = raw_eval.map(lambda ex:  {"text": format_example(ex)})

print(f"\nTrain size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ── Model (4-bit QLoRA) ───────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

model.config.use_cache = False

# ── LoRA ──────────────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── Collator ──────────────────────────────────────────────────────────────────
collator = CompletionOnlyCollator(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

# ── Training args ─────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    max_length=MAX_LENGTH,
    packing=False,

    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=52,

    num_train_epochs=5,

    fp16=False,
    bf16=True,

    eval_strategy="steps",
    eval_steps=50,

    logging_steps=10,
    save_steps=100,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,

    dataloader_num_workers=8,
    dataloader_pin_memory=True,

    report_to="wandb",
    run_name="qwen2.5-7b-lol-commentary-G4",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    formatting_func=lambda ex: ex["text"],   # string not list
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)


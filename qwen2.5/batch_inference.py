# BATCH TEST INFERENCE -7B FULL - 30s 600 max toks - 800 max toks 60s

import json
import torch
from pathlib import Path
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR      = "/content/drive/MyDrive/MLP/outputs"
TEST_FILE        = "/content/drive/MyDrive/MLP/test.jsonl"
OUTPUT_DIR       = Path("/content/drive/MyDrive/MLP/results/qwen7b_finetuned")
GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"
BATCH_SIZE       = 32  # 96GB VRAM — push this up, reduce if you hit OOM

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"bf16: {torch.cuda.is_bf16_supported()}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # required for batched generation

# ── Model (full bfloat16 — no quantization needed on 96GB) ───────────────────
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# ── Load test data ────────────────────────────────────────────────────────────
test_dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]
examples     = list(test_dataset)
print(f"Total test samples: {len(examples)}")

# ── Resume: skip already-generated inputs ────────────────────────────────────
saved_inputs = set()
if GENERATIONS_FILE.exists():
    with open(GENERATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            saved_inputs.add(ex["input"])
    print(f"Already generated: {len(saved_inputs)} — resuming from there")
else:
    print("No existing generations found — starting fresh")

remaining = [ex for ex in examples if ex["input"] not in saved_inputs]
print(f"Remaining to generate: {len(remaining)}")

# ── Batch generate ────────────────────────────────────────────────────────────
def generate_batch(input_texts, max_new_tokens=400):
    prompts = [
        "### Task:\n"
        "Generate live League of Legends shoutcaster commentary from the event log.\n\n"
        "### Input:\n"
        f"{text}\n\n"
        "### Response:\n"
        for text in input_texts
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=800,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        for output in outputs
    ]

# ── Run and append ────────────────────────────────────────────────────────────
with open(GENERATIONS_FILE, "a", encoding="utf-8") as f:
    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Generating"):
        batch           = remaining[i:i + BATCH_SIZE]
        input_texts     = [ex["input"] for ex in batch]
        generated_texts = generate_batch(input_texts)
        for ex, prediction in zip(batch, generated_texts):
            result = {
                "input":      ex["input"],
                "target":     ex["target"],
                "prediction": prediction,
                "metadata":   {}
            }
            f.write(json.dumps(result) + "\n")
        f.flush()   # guard against mid-run disconnects

# ── Verify ────────────────────────────────────────────────────────────────────
total = sum(1 for _ in open(GENERATIONS_FILE))
print(f"Total generations saved: {total}")

# ── Auto disconnect ───────────────────────────────────────────────────────────
from google.colab import runtime
runtime.unassign()
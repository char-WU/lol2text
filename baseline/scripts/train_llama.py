import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "meta-llama/Llama-2-13b-hf"
HF_DATASET_NAME = "your-username/your-lol-dataset-name" 
OUTPUT_DIR = "./models/llama13b-base_LoL19"

MAX_STEPS = 10000
LEARNING_RATE = 1e-3
LORA_DROPOUT = 0.1
dataset = load_dataset(HF_DATASET_NAME)
train_data = dataset["train"]
eval_data = dataset["validation"]

# Initialize Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def formatting_prompts_func(example):
  output_texts = []
  # Handle batch processing from SFTTrainer
  if isinstance(example['input'], list):
    for i in range(len(example['input'])):
      text = (
        "You are an expert of League of Legends esports games. "
        "Please read the input data records and describe them in natural language commentary as output.\n"
        f"Input: {example['input'][i]}\n"
        f"Output: {example['target'][i]}{tokenizer.eos_token}"
      )
      output_texts.append(text)
  else:
    text = (
      "You are an expert of League of Legends esports games. "
      "Please read the input data records and describe them in natural language commentary as output.\n"
      f"Input: {example['input']}\n"
      f"Output: {example['target']}{tokenizer.eos_token}"
    )
    output_texts.append(text)
  return output_texts

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16
)

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
  MODEL_ID,
  quantization_config=bnb_config,
  device_map="auto"
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  max_steps=MAX_STEPS,
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  gradient_accumulation_steps=16,
  optim="paged_adamw_32bit",
  evaluation_strategy="steps",
  eval_steps=500,
  save_steps=1000,
  logging_steps=50,
  fp16=True,
  max_grad_norm=0.3,
  warmup_ratio=0.03,
  lr_scheduler_type="constant",
  report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting training...")
trainer.train()

trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print(f"Training complete! Adapter saved to {OUTPUT_DIR}/final_adapter")
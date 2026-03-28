import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import argparse
import numpy as np
import os
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  get_linear_schedule_with_warmup,
  set_seed
)

from tqdm import tqdm

logging.basicConfig(
  format = '%(asctime)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

def generate_output_dir(
    model_name: str,
    dataset_name: Optional[str] = None
) -> Path:
  """
  1. model_name + dataset_name → results/{model_name}_{dataset_name}/
  2. model_name + train_file → results/{model_name}_{extracted_name}/
  3. model_name only → results/{model_name}/
  
  Examples:
  - model_name="t5-base", dataset_name="LoL19" 
    → results/t5-base_LoL19/
  - model_name="facebook/bart-base", train_file="data/merged/LoL1921_train.jsonl"
    → results/bart-base_LoL1921/
  """
  if '/' in model_name:
      model_short = model_name.split('/')[-1]
  else:
    model_short = model_name
  parts = [model_short]
  
  if dataset_name:
    if '/' in dataset_name:
      dataset_short = dataset_name.split('/')[-1]
    else:
      dataset_short = dataset_name
    parts.append(dataset_short)
  
  dir_name = '_'.join(parts)
  output_dir = Path('results') / dir_name
  
  if output_dir.exists():
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / f"{dir_name}_{timestamp}"
  
  return output_dir

@dataclass
class TrainingConfig:
  model_name: str = "t5-base"
  max_input_length: int = 512
  max_target_length: int = 512
  batch_size: int = 8
  learning_rate: float = 1e-3 # 5e-5
  num_epochs: int = 100 # upper limit
  max_steps: int = 10000
  warmup_steps: int = 500
  gradient_accumulation_steps: int = 1
  max_grad_norm: float = 1.0
  dropout_rate: float = 0.5
  eval_steps: int = 500
  save_steps: int = 1000
  logging_steps: int = 100
  seed: int = 42

class Dataset(Dataset):
  """
  Dataset for commentary generation
  """
  def __init__(
      self,
      data_source,
      tokenizer,
      max_input_length: int = 512,
      max_target_length: int = 512,
      is_hf_dataset: bool = False
  ):
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    
    if is_hf_dataset:
      self.examples = data_source
    else:
      self.examples = self._load_jsonl(data_source)
    logger.info(f"Loaded {len(self.examples)} examples")

  def _load_jsonl(self, filepath):
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          example = json.loads(line)
          if 'input' in example and 'target' in example:
            examples.append(example)
        except json.JSONDecodeError:
          continue
    return examples

  def __len__(self):
    return len(self.examples)
  
  def __getitem__(self, idx):
    example = self.examples[idx]
    
    input_text = example['input']
    target_text = example['target']
    
    input_encoding = self.tokenizer(
      input_text,
      max_length=self.max_input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
    target_encoding = self.tokenizer(
      target_text,
      max_length=self.max_target_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
    
    labels = target_encoding['input_ids'].squeeze()
    # set padding token = -100
    labels[labels == self.tokenizer.pad_token_id] = -100
    
    return {
      'input_ids': input_encoding['input_ids'].squeeze(),
      'attention_mask': input_encoding['attention_mask'].squeeze(),
      'labels': labels
    }

def load_dataset_from_hf(dataset_name: str, split: str = 'train'):
  """    
  Args:
    dataset_name: HuggingFace dataset name (e.g., "your-username/LoL19")
    split: 'train', 'validation', or 'test'
  """
  try:
    from datasets import load_dataset
    logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset
  except ImportError:
    raise ImportError("Please install datasets: pip install datasets")
  except Exception as e:
    raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

def dataloaders(
    config: TrainingConfig,
    tokenizer,
    dataset_name: Optional[str] = None
):
  if dataset_name:
    logger.info(f"Using HuggingFace dataset: {dataset_name}")
    train_hf = load_dataset_from_hf(dataset_name, split='train')
    val_hf = load_dataset_from_hf(dataset_name, split='validation')

    # shuffle the dataset
    logger.info(f"Shuffling training dataset with seed {config.seed}")
    train_hf = train_hf.shuffle(seed=config.seed)
    
    train_dataset = Dataset(
      train_hf,
      tokenizer,
      config.max_input_length,
      config.max_target_length,
      is_hf_dataset=True
    )
    val_dataset = Dataset(
      val_hf,
      tokenizer,
      config.max_input_length,
      config.max_target_length,
      is_hf_dataset=True
    )
  else:
    train_file, val_file = get_local_dataset(dataset_name)
    logger.info(f"Loading from local files:")
    logger.info(f"  Train: {train_file}")
    logger.info(f"  Val:   {val_file}")
    logger.info(f"Using local files: {train_file}, {val_file}")
    train_dataset = Dataset(
      train_file,
      tokenizer,
      config.max_input_length,
      config.max_target_length,
      is_hf_dataset=False
    )
    val_dataset = Dataset(
      val_file,
      tokenizer,
      config.max_input_length,
      config.max_target_length,
      is_hf_dataset=False
    )

  train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True
  )
  
  return train_loader, val_loader

def evaluate(model, val_loader, device):
  model.eval()
  total_loss = 0
  num_batches = 0
  
  with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
      )
      
      total_loss += outputs.loss.item()
      num_batches += 1
  
  avg_loss = total_loss / num_batches if num_batches > 0 else 0
  perplexity = np.exp(avg_loss)
  
  return {
    'loss': avg_loss,
    'perplexity': perplexity
  }

def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, output_dir, is_best=False):
  output_dir = Path(output_dir)
  
  if is_best:
    checkpoint_dir = output_dir / "best_model"
  else:
    checkpoint_dir = output_dir / f"checkpoint-epoch{epoch}-step{step}"
  
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  
  model.save_pretrained(checkpoint_dir)
  tokenizer.save_pretrained(checkpoint_dir)
  
  torch.save({
    'epoch': epoch,
    'step': step,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
  }, checkpoint_dir / 'training_state.pt')
  
  logger.info(f"✓ Saved checkpoint to {checkpoint_dir}")

def train(
    config: TrainingConfig,
    model,
    tokenizer,
    train_loader,
    val_loader,
    output_dir: str,
    resume_from: Optional[str] = None
):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f"Using device: {device}")
  model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
  total_steps = config.max_steps
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=total_steps
  )
  
  # Resume from checkpoint
  start_epoch = 0
  global_step = 0
  
  if resume_from:
    logger.info(f"Resuming from checkpoint: {resume_from}")
    checkpoint = torch.load(Path(resume_from) / 'training_state.pt')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['step']
  
  # TensorBoard
  writer = SummaryWriter(log_dir=Path(output_dir) / 'runs')
  
  best_val_loss = float('inf')
  for epoch in range(start_epoch, config.num_epochs):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for step, batch in enumerate(progress_bar):
      if global_step >= config.max_steps:
        logger.info(f"\n✓ Reached max_steps ({config.max_steps}), stopping training")
        break

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
      )
      
      loss = outputs.loss / config.gradient_accumulation_steps
      loss.backward()
      epoch_loss += loss.item()
      
      # Gradient accumulation
      if (step + 1) % config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        
        if global_step % config.logging_steps == 0:
          avg_loss = epoch_loss / (step + 1)
          writer.add_scalar('train/loss', avg_loss, global_step)
          writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
          progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Evaluation
        if global_step % config.eval_steps == 0:
            logger.info(f"\n{'='*60}\nEvaluating at step {global_step}\n{'='*60}")
            val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar('val/loss', val_metrics['loss'], global_step)
            writer.add_scalar('val/perplexity', val_metrics['perplexity'], global_step)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
              best_val_loss = val_metrics['loss']
              save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, global_step, output_dir, is_best=True)
              logger.info(f"🏆 New best model! Val loss: {best_val_loss:.4f}")
            model.train()
        
        # Save checkpoint
        if global_step % config.save_steps == 0:
          save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, global_step, output_dir, is_best=False)
    
    # End of epoch
    logger.info(f"\nEpoch {epoch+1} completed. Avg loss: {epoch_loss / len(train_loader):.4f}")
    if global_step >= config.max_steps:
      break
  
  writer.close()
  logger.info(f"\n✓ Training completed! Best model saved to {output_dir}/best_model")

def get_local_dataset(name: str, base_dir: str = 'split') -> tuple:
  dataset_dir = Path(base_dir) / name
  train_file = dataset_dir / 'train.jsonl'
  val_file = dataset_dir / 'val.jsonl'
  if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")
  if not val_file.exists():
    raise FileNotFoundError(f"Validation file not found: {val_file}")
  return str(train_file), str(val_file)

def main():
  parser = argparse.ArgumentParser(description="Train baseline commentary generation model")
  
  # Model
  parser.add_argument('--model-name', type=str, default='t5-base',
                      help='Pretrained model name (t5-base, t5-large etc.)')
  # Data source (choose one)
  parser.add_argument('--name', type=str, default=None,
                      help='HuggingFace dataset name (e.g., "your-username/LoL19")')
  # Output
  parser.add_argument('--resume-from', type=str, default=None,
                      help='Resume training from checkpoint directory')
  
  # Training hyperparameters
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--learning-rate', type=float, default=5e-5)
  parser.add_argument('--num-epochs', type=int, default=10)
  parser.add_argument('--max-input-length', type=int, default=512)
  parser.add_argument('--max-target-length', type=int, default=256)
  parser.add_argument('--warmup-steps', type=int, default=500)
  parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
  parser.add_argument('--eval-steps', type=int, default=500)
  parser.add_argument('--save-steps', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=42)
  
  args = parser.parse_args()
  
  # Validate inputs
  if not args.name:
    raise ValueError("Must provide either --dataset-name")
  
  # Set seed
  set_seed(args.seed)
  
  # Create config
  config = TrainingConfig(
    model_name=args.model_name,
    max_input_length=args.max_input_length,
    max_target_length=args.max_target_length,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    num_epochs=args.num_epochs,
    warmup_steps=args.warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    seed=args.seed
  )

  output_dir = generate_output_dir(
    model_name=args.model_name,
    dataset_name=args.name
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  
  logger.info("=" * 60)
  logger.info("TRAINING CONFIGURATION")
  logger.info(f"Model: {config.model_name}")
  logger.info(f"Batch size: {config.batch_size}")
  logger.info(f"Learning rate: {config.learning_rate}")
  logger.info(f"Dropout rate: {config.dropout_rate} (paper: 0.5)")
  logger.info(f"Max steps: {config.max_steps} (paper: 10,000)")
  logger.info(f"Epochs: {config.num_epochs}")
  logger.info(f"Output dir: {output_dir}")
  
  logger.info(f"Dataset: {args.name} (HuggingFace)")
  
  # Load tokenizer and model
  logger.info("\nLoading tokenizer and model...")
  tokenizer = AutoTokenizer.from_pretrained(config.model_name)
  
  # 1. Load the model with default config (standard 0.1 dropout everywhere)
  model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
  
  # 2. Force ONLY the decoder dropout modules to 0.5
  if hasattr(model, 'decoder'):
    for name, module in model.decoder.named_modules():
      if isinstance(module, torch.nn.Dropout):
        module.p = config.dropout_rate
  
  logger.info(f"✓ Applied {config.dropout_rate} dropout exclusively to the decoder")
  
  logger.info(f"✓ Loaded {config.model_name}")
  logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
  
  # Create dataloaders
  logger.info("\nPreparing datasets...")
  train_loader, val_loader = dataloaders(
    config,
    tokenizer,
    dataset_name=args.name
  )
  
  logger.info(f"✓ Train batches: {len(train_loader)}")
  logger.info(f"✓ Val batches: {len(val_loader)}")
  
  # Train
  logger.info("\n" + "=" * 60)
  logger.info("STARTING TRAINING")
  
  train(
    config,
    model,
    tokenizer,
    train_loader,
    val_loader,
    str(output_dir),
    resume_from=args.resume_from
  )

 
if __name__ == '__main__':
  main()
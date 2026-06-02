"""Training script for NietzscheBot GPT-2 fine-tuning."""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Optional

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.config import TrainingConfig
from src.training.dataset import create_datasets
from src.training.utils import set_seed, setup_logging, count_parameters, format_time

logger = logging.getLogger(__name__)


def train(config: TrainingConfig, corpus_path: Path, resume_from_checkpoint: Optional[str] = None) -> None:
    """
    Train GPT-2 or Gemma 3 model on Nietzsche corpus.

    Args:
        config: Training configuration
        corpus_path: Path to training_corpus.txt
        resume_from_checkpoint: Path to checkpoint dir to resume from, or "true" to auto-detect latest, or None
    """
    start_time = time.time()

    # Setup
    set_seed(config.seed)
    setup_logging(Path(config.logging_dir))

    logger.info("=" * 80)
    logger.info("NietzscheBot Training")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info("=" * 80)

    # Quantization config for 4-bit loading
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer and model
    logger.info(f"Loading {config.model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if config.load_in_4bit else None,
        torch_dtype=torch.bfloat16 if config.load_in_4bit else None,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA if configured
    if config.use_lora:
        logger.info("Applying LoRA configuration...")
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Model info
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        train_test_split=config.train_test_split,
        seed=config.seed
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(val_dataset)}")

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )

    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use LoRA-specific learning rate if LoRA is enabled
    effective_learning_rate = config.lora_learning_rate if config.use_lora else config.learning_rate

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=effective_learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        adam_epsilon=config.adam_epsilon,
        warmup_ratio=config.warmup_ratio,
        logging_dir=config.logging_dir,
        logging_steps=100,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.use_lora,  # Memory efficiency when using LoRA
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # Train
    logger.info("Starting training...")
    # Handle resume_from_checkpoint: if "true", pass True for auto-detect; if path, pass path; if None, pass None
    resume_checkpoint = None
    if resume_from_checkpoint is not None:
        if resume_from_checkpoint.lower() == "true":
            resume_checkpoint = True
        else:
            resume_checkpoint = resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    final_model_path = output_dir / "final"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save config
    config.save(output_dir / "training_config.json")

    # Training complete
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Training complete in {format_time(elapsed_time)}")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train NietzscheBot GPT-2 model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config JSON file"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="src/data/processed/training_corpus.txt",
        help="Path to training corpus"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2, gpt2-medium, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for finetuning"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from, or 'true' to auto-detect latest"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TrainingConfig.load(Path(args.config))
    else:
        config = TrainingConfig()

    # Override with CLI arguments if provided
    if args.model:
        config.model_name = args.model
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.use_lora:
        config.use_lora = True
    if args.load_in_4bit:
        config.load_in_4bit = True

    # Resolve corpus path
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Train
    train(config, corpus_path, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()

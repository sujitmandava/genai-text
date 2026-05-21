"""Training script for NietzscheBot GPT-2 fine-tuning."""

import argparse
import logging
from pathlib import Path
import time

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from .config import TrainingConfig
from .dataset import create_datasets
from .utils import set_seed, get_device, setup_logging, count_parameters, format_time

logger = logging.getLogger(__name__)


def train(config: TrainingConfig, corpus_path: Path) -> None:
    """
    Train GPT-2 model on Nietzsche corpus.

    Args:
        config: Training configuration
        corpus_path: Path to training_corpus.txt
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

    # Load tokenizer and model
    logger.info(f"Loading {config.model_name} tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    model = GPT2LMHeadModel.from_pretrained(config.model_name)

    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

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

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        adam_epsilon=config.adam_epsilon,
        warmup_steps=config.warmup_steps,
        logging_dir=config.logging_dir,
        logging_steps=100,
        save_steps=config.checkpoint_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=config.seed,
        fp16=torch.cuda.is_available(),  # Mixed precision if CUDA available
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

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

    # Resolve corpus path
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Train
    train(config, corpus_path)


if __name__ == "__main__":
    main()

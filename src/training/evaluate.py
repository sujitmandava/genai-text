"""Evaluation script for NietzscheBot model."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.config import TrainingConfig
from src.training.dataset import NietzscheDataset
from src.training.utils import set_seed, get_device

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: GPT2LMHeadModel,
    dataset: NietzscheDataset,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: Fine-tuned GPT-2 model
        dataset: Validation dataset
        batch_size: Batch size for evaluation
        device: Device to use (defaults to auto-detect)

    Returns:
        Perplexity score
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    logger.info(f"Computing perplexity on {len(dataset)} examples...")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Accumulate loss
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")

    return perplexity


def generate_samples(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompts: list[str],
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: Optional[torch.device] = None
) -> list[dict[str, str]]:
    """
    Generate text samples from prompts.

    Args:
        model: Fine-tuned GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompts: List of prompt strings
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Device to use

    Returns:
        List of dicts with 'prompt' and 'generated_text'
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    samples = []

    logger.info(f"Generating {len(prompts)} samples...")

    for prompt in prompts:
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        samples.append({
            'prompt': prompt,
            'generated_text': generated_text
        })

        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text}\n")

    return samples


def evaluate(
    checkpoint_path: Path,
    corpus_path: Path,
    config: Optional[TrainingConfig] = None,
    output_path: Optional[Path] = None
) -> dict:
    """
    Run full evaluation: perplexity + generation samples.

    Args:
        checkpoint_path: Path to model checkpoint directory
        corpus_path: Path to training_corpus.txt
        config: Training config (optional, will look for saved config)
        output_path: Path to save evaluation report JSON

    Returns:
        Evaluation report dictionary
    """
    logger.info("=" * 80)
    logger.info("NietzscheBot Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Load config if not provided
    if config is None:
        config_path = checkpoint_path.parent / "training_config.json"
        if config_path.exists():
            config = TrainingConfig.load(config_path)
        else:
            logger.warning("No training config found, using defaults")
            config = TrainingConfig()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

    device = get_device()

    # Create validation dataset
    logger.info("Loading validation dataset...")
    val_dataset = NietzscheDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        train_split=False,
        train_test_split=config.train_test_split,
        seed=config.seed
    )

    # Compute perplexity
    perplexity = compute_perplexity(
        model=model,
        dataset=val_dataset,
        batch_size=config.batch_size,
        device=device
    )

    # Generate samples
    prompts = [
        "God is",
        "Truth",
        "The will to power",
        "What does not kill me",
        "Man is something that shall be overcome",
        "There are no facts,",
        "The overman",
        "Morality is"
    ]

    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=config.generation_max_length,
        temperature=config.generation_temperature,
        top_p=config.generation_top_p,
        top_k=config.generation_top_k,
        device=device
    )

    # Build report
    report = {
        'checkpoint': str(checkpoint_path),
        'model_name': config.model_name,
        'perplexity': perplexity,
        'validation_size': len(val_dataset),
        'generation_samples': samples,
        'generation_config': {
            'max_length': config.generation_max_length,
            'temperature': config.generation_temperature,
            'top_p': config.generation_top_p,
            'top_k': config.generation_top_k
        }
    }

    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to: {output_path}")

    logger.info("=" * 80)
    logger.info("Evaluation complete")
    logger.info(f"Perplexity: {perplexity:.2f}")
    logger.info("=" * 80)

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate NietzscheBot model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="src/data/processed/training_corpus.txt",
        help="Path to training corpus"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation report"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    output_path = Path(args.output)

    # Evaluate
    evaluate(
        checkpoint_path=checkpoint_path,
        corpus_path=corpus_path,
        output_path=output_path
    )


if __name__ == "__main__":
    main()

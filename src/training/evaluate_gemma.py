"""Evaluation script for Gemma-LoRA fine-tuned model."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.config import TrainingConfig
from src.training.dataset import NietzscheDataset
from src.training.utils import set_seed, get_device

logger = logging.getLogger(__name__)


def load_model_with_lora(
    base_model_name: str,
    lora_checkpoint: Path,
    load_in_4bit: bool = True
):
    """
    Load base model and apply LoRA adapter.

    Args:
        base_model_name: HuggingFace model name (e.g., google/gemma-3-4b-it)
        lora_checkpoint: Path to LoRA adapter checkpoint
        load_in_4bit: Whether to load in 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading LoRA adapter from: {lora_checkpoint}")
    model = PeftModel.from_pretrained(model, lora_checkpoint)

    return model, tokenizer


def compute_perplexity(
    model,
    dataset: NietzscheDataset,
    batch_size: int = 4,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: Model with LoRA adapter
        dataset: Validation dataset
        batch_size: Batch size for evaluation
        device: Device to use (defaults to auto-detect)

    Returns:
        Perplexity score
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    logger.info(f"Computing perplexity on {len(dataset)} examples...")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")

    return perplexity


def generate_samples(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
) -> list[dict[str, str]]:
    """
    Generate text samples from prompts.

    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter

    Returns:
        List of dicts with 'prompt' and 'generated_text'
    """
    model.eval()
    samples = []

    logger.info(f"Generating {len(prompts)} samples...")

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        samples.append({
            'prompt': prompt,
            'generated_text': generated_text
        })

        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text}\n")

    return samples


def evaluate(
    base_model_name: str,
    lora_checkpoint: Path,
    corpus_path: Path,
    output_path: Optional[Path] = None,
    load_in_4bit: bool = True,
) -> dict:
    """
    Run full evaluation: perplexity + generation samples.

    Args:
        base_model_name: HuggingFace model name
        lora_checkpoint: Path to LoRA adapter checkpoint
        corpus_path: Path to training_corpus.txt
        output_path: Path to save evaluation report JSON
        load_in_4bit: Whether to load in 4-bit quantization

    Returns:
        Evaluation report dictionary
    """
    logger.info("=" * 80)
    logger.info("Gemma-LoRA Evaluation")
    logger.info("=" * 80)
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"LoRA checkpoint: {lora_checkpoint}")

    config_path = lora_checkpoint.parent / "training_config.json"
    if config_path.exists():
        config = TrainingConfig.load(config_path)
    else:
        logger.warning("No training config found, using defaults")
        config = TrainingConfig()

    set_seed(config.seed)

    model, tokenizer = load_model_with_lora(
        base_model_name=base_model_name,
        lora_checkpoint=lora_checkpoint,
        load_in_4bit=load_in_4bit,
    )

    logger.info("Loading validation dataset...")
    val_dataset = NietzscheDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        train_split=False,
        train_test_split=config.train_test_split,
        seed=config.seed
    )

    perplexity = compute_perplexity(
        model=model,
        dataset=val_dataset,
        batch_size=config.batch_size,
    )

    prompts = [
        "What is the will to power?",
        "Explain the concept of the overman.",
        "What does Nietzsche say about truth?",
        "How does one become what one is?",
        "What is the meaning of eternal recurrence?",
        "Why did God die?",
        "What makes a person great?",
        "How should we live?",
    ]

    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=config.generation_max_length,
        temperature=config.generation_temperature,
        top_p=config.generation_top_p,
        top_k=config.generation_top_k,
    )

    report = {
        'base_model': base_model_name,
        'lora_checkpoint': str(lora_checkpoint),
        'perplexity': perplexity,
        'validation_size': len(val_dataset),
        'generation_samples': samples,
        'generation_config': {
            'max_new_tokens': config.generation_max_length,
            'temperature': config.generation_temperature,
            'top_p': config.generation_top_p,
            'top_k': config.generation_top_k
        }
    }

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
    parser = argparse.ArgumentParser(description="Evaluate Gemma-LoRA model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint directory"
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
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full precision)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    lora_checkpoint = Path(args.lora_checkpoint)
    if not lora_checkpoint.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint}")

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    output_path = Path(args.output)

    evaluate(
        base_model_name=args.base_model,
        lora_checkpoint=lora_checkpoint,
        corpus_path=corpus_path,
        output_path=output_path,
        load_in_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()

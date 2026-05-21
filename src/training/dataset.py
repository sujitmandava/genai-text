"""Dataset preparation for NietzscheBot training."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


class NietzscheDataset(Dataset):
    """
    PyTorch Dataset for Nietzsche corpus fine-tuning.

    Loads text, tokenizes, and chunks into fixed-length blocks.
    """

    def __init__(
        self,
        corpus_path: Path,
        tokenizer: GPT2Tokenizer,
        max_seq_length: int = 512,
        train_split: bool = True,
        train_test_split: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize dataset.

        Args:
            corpus_path: Path to training_corpus.txt
            tokenizer: GPT-2 tokenizer instance
            max_seq_length: Maximum sequence length for each example
            train_split: If True, use training split; else validation split
            train_test_split: Fraction of data to use for validation (default 0.1 = 10%)
            seed: Random seed for reproducible splits
        """
        self.corpus_path = Path(corpus_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.seed = seed

        logger.info(f"Loading corpus from {self.corpus_path}")

        # Load and tokenize entire corpus
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"Corpus size: {len(text):,} characters")

        # Tokenize entire corpus
        tokenized = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=False
        )
        input_ids = tokenized['input_ids'].squeeze()

        logger.info(f"Total tokens: {len(input_ids):,}")

        # Chunk into blocks of max_seq_length
        # Each block is a training example
        self.examples = []

        for i in range(0, len(input_ids) - max_seq_length + 1, max_seq_length):
            chunk = input_ids[i:i + max_seq_length]
            if len(chunk) == max_seq_length:
                self.examples.append(chunk)

        logger.info(f"Created {len(self.examples):,} chunks of length {max_seq_length}")

        # Split into train/validation
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.examples))
        split_idx = int(len(self.examples) * (1 - train_test_split))

        if train_split:
            selected_indices = indices[:split_idx]
            logger.info(f"Using training split: {len(selected_indices):,} examples")
        else:
            selected_indices = indices[split_idx:]
            logger.info(f"Using validation split: {len(selected_indices):,} examples")

        self.examples = [self.examples[i] for i in selected_indices]

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        For language modeling, input_ids and labels are the same.

        Args:
            idx: Index of example

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        input_ids = self.examples[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'labels': input_ids  # For causal LM, labels = input_ids
        }


def create_datasets(
    corpus_path: Path,
    tokenizer: GPT2Tokenizer,
    max_seq_length: int = 512,
    train_test_split: float = 0.1,
    seed: int = 42
) -> tuple[NietzscheDataset, NietzscheDataset]:
    """
    Create train and validation datasets.

    Args:
        corpus_path: Path to training_corpus.txt
        tokenizer: GPT-2 tokenizer
        max_seq_length: Maximum sequence length
        train_test_split: Fraction for validation split
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = NietzscheDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        train_split=True,
        train_test_split=train_test_split,
        seed=seed
    )

    val_dataset = NietzscheDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        train_split=False,
        train_test_split=train_test_split,
        seed=seed
    )

    return train_dataset, val_dataset

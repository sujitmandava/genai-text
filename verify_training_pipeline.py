#!/usr/bin/env python
"""
Verification script for training pipeline.
Tests all components without running full training.
"""

import sys
from pathlib import Path

print("="*80)
print("NietzscheBot Training Pipeline Verification")
print("="*80)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from src.training import (
        TrainingConfig,
        NietzscheDataset,
        create_datasets,
        set_seed,
        get_device,
        setup_logging,
        count_parameters,
        format_time
    )
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Config
print("\n[2/5] Testing TrainingConfig...")
try:
    config = TrainingConfig(epochs=1, batch_size=2)
    temp_config_path = Path("temp_test_config.json")
    config.save(temp_config_path)
    loaded = TrainingConfig.load(temp_config_path)
    temp_config_path.unlink()
    assert loaded.epochs == 1
    assert loaded.batch_size == 2
    print(f"  ✓ Config save/load works")
    print(f"    Model: {config.model_name}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Max seq length: {config.max_seq_length}")
except Exception as e:
    print(f"  ✗ Config test failed: {e}")
    sys.exit(1)

# Test 3: Dataset
print("\n[3/5] Testing NietzscheDataset...")
try:
    from transformers import GPT2Tokenizer

    corpus_path = Path("src/data/processed/training_corpus.txt")
    if not corpus_path.exists():
        print(f"  ✗ Corpus not found at {corpus_path}")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = create_datasets(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_length=128,
        train_test_split=0.1,
        seed=42
    )

    print(f"  ✓ Dataset creation successful")
    print(f"    Train examples: {len(train_ds)}")
    print(f"    Val examples: {len(val_ds)}")

    # Test getting an example
    example = train_ds[0]
    assert 'input_ids' in example
    assert 'attention_mask' in example
    assert 'labels' in example
    print(f"    Example shape: {example['input_ids'].shape}")

except Exception as e:
    print(f"  ✗ Dataset test failed: {e}")
    sys.exit(1)

# Test 4: Model loading
print("\n[4/5] Testing model loading...")
try:
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    params = count_parameters(model)
    device = get_device()

    print(f"  ✓ Model loaded successfully")
    print(f"    Parameters: {params['total']:,}")
    print(f"    Device: {device}")

except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    sys.exit(1)

# Test 5: Utilities
print("\n[5/5] Testing utilities...")
try:
    set_seed(42)
    time_str = format_time(3661)  # 1h 1m 1s
    assert "1h" in time_str
    print(f"  ✓ Utilities working")
    print(f"    format_time(3661) = '{time_str}'")

except Exception as e:
    print(f"  ✗ Utilities test failed: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "="*80)
print("SUCCESS: All pipeline components verified!")
print("="*80)
print("\nTo train a model, run:")
print("  python -m src.training.train")
print("\nFor custom training:")
print("  python -m src.training.train --model gpt2 --epochs 3 --batch-size 4")
print("\nTo evaluate a trained model:")
print("  python -m src.training.evaluate --checkpoint models/nietzsche-gpt2/final")
print("="*80)

# NietzscheBot Training Pipeline

GPT-2 fine-tuning pipeline for Nietzsche corpus.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Training

Train with default settings (gpt2-small, 3 epochs):

```bash
python -m src.training.train
```

### 3. Custom Training

```bash
python -m src.training.train \
  --model gpt2-medium \
  --epochs 5 \
  --batch-size 4 \
  --output-dir models/nietzsche-gpt2-medium
```

### 4. Using a Config File

```python
from src.training import TrainingConfig

config = TrainingConfig(
    model_name="gpt2-medium",
    batch_size=4,
    epochs=5,
    max_seq_length=512,
    output_dir="models/my-nietzsche-model"
)
config.save("my_config.json")
```

Then train:

```bash
python -m src.training.train --config my_config.json
```

## Evaluation

Evaluate a trained model:

```bash
python -m src.training.evaluate \
  --checkpoint models/nietzsche-gpt2/final \
  --output evaluation_report.json
```

This will:
- Compute perplexity on validation set
- Generate sample texts from prompts
- Save results to `evaluation_report.json`

## Module Overview

### `config.py`
- `TrainingConfig` dataclass with all hyperparameters
- JSON export/import for reproducibility

### `dataset.py`
- `NietzscheDataset`: PyTorch Dataset for corpus
- Tokenization with GPT-2 tokenizer
- 90/10 train/val split with fixed seed

### `train.py`
- Training script using HuggingFace Trainer
- Automatic checkpointing every 500 steps
- TensorBoard logging
- CLI interface

### `evaluate.py`
- Perplexity computation
- Text generation with custom prompts
- JSON report export

### `utils.py`
- `set_seed()`: Reproducibility
- `get_device()`: CUDA/MPS/CPU detection
- `setup_logging()`: Logging configuration
- `count_parameters()`: Model parameter counting

## Training Configuration

Default hyperparameters:

```python
model_name = "gpt2"
batch_size = 4
learning_rate = 5e-5
epochs = 3
max_seq_length = 512
gradient_accumulation_steps = 4
warmup_steps = 500
```

## Output Structure

```
models/nietzsche-gpt2/
├── checkpoint-500/
├── checkpoint-1000/
├── final/                     # Final trained model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
└── training_config.json       # Training config for reproducibility

logs/training/
├── training.log
└── events.out.tfevents...     # TensorBoard logs
```

## TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir logs/training
```

## Example: Full Training Run

```python
from pathlib import Path
from src.training import TrainingConfig, train

config = TrainingConfig(
    model_name="gpt2",
    epochs=3,
    batch_size=4,
    max_seq_length=512,
    output_dir="models/nietzsche-gpt2"
)

corpus_path = Path("src/data/processed/training_corpus.txt")
train(config, corpus_path)
```

## Hardware Requirements

- **gpt2 (small)**: ~500MB VRAM, ~2-3 hours on M1/M2 Mac
- **gpt2-medium**: ~1.5GB VRAM, ~4-5 hours on M1/M2 Mac
- **gpt2-large**: ~3GB VRAM, ~8-10 hours on M1/M2 Mac

Mixed precision (FP16) is automatically enabled on CUDA GPUs.

## Troubleshooting

### Out of Memory
Reduce batch size or max_seq_length:

```python
config = TrainingConfig(
    batch_size=2,
    max_seq_length=256
)
```

### Slow Training
Increase gradient accumulation:

```python
config = TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=8  # Effective batch = 2*8 = 16
)
```

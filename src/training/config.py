"""Training configuration for NietzscheBot."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for GPT-2 or Gemma 3 fine-tuning on Nietzsche corpus."""

    # Model
    model_name: str = "gpt2"

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 5e-5
    epochs: int = 3
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500

    # Paths
    output_dir: str = "models/nietzsche-gpt2"
    logging_dir: str = "logs/training"

    # Checkpointing
    checkpoint_steps: int = 500
    save_total_limit: int = 3

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_epsilon: float = 1e-8

    # Data
    train_test_split: float = 0.1
    seed: int = 42

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"

    # Quantization
    load_in_4bit: bool = False

    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: int = 1

    # Generation settings for evaluation
    generation_max_length: int = 100
    generation_temperature: float = 0.8
    generation_top_p: float = 0.9
    generation_top_k: int = 50

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        assert 0 < self.train_test_split < 1, "train_test_split must be between 0 and 1"

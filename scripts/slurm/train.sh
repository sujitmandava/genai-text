#!/usr/bin/env bash
#SBATCH --job-name=text-train
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/slurm/text-train-%j.out
#SBATCH --error=logs/slurm/text-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Train GPT-2 model on Nietzsche corpus for genai-text project
# GPU job with configurable hyperparameters via environment variables

set -euo pipefail

# Print job context
echo "======================================================================"
echo "Job: train GPT-2 model"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Started: $(date)"
echo "======================================================================"

# Derive repo root from script location (portable across clusters)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
echo "Working directory: ${REPO_ROOT}"

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Environment activation (optional, env-driven)
if [[ -n "${GENAI_TEXT_ENV:-}" ]]; then
    if [[ -f "${GENAI_TEXT_ENV}" ]]; then
        echo "Activating environment from file: ${GENAI_TEXT_ENV}"
        source "${GENAI_TEXT_ENV}"
    elif command -v conda &> /dev/null; then
        echo "Activating conda environment: ${GENAI_TEXT_ENV}"
        eval "$(conda shell.bash hook)"
        conda activate "${GENAI_TEXT_ENV}"
    else
        echo "WARNING: GENAI_TEXT_ENV is set but conda is not available"
    fi
elif [[ -f .venv/bin/activate ]]; then
    echo "Activating local virtualenv: .venv"
    source .venv/bin/activate
fi

# Training parameters (env-overridable with conservative defaults)
TRAIN_MODEL="${TRAIN_MODEL:-gpt2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-models/nietzsche-bot}"
TRAIN_CORPUS="${TRAIN_CORPUS:-src/data/processed/training_corpus.txt}"

echo ""
echo "Training configuration:"
echo "  Model: ${TRAIN_MODEL}"
echo "  Epochs: ${TRAIN_EPOCHS}"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  Output directory: ${TRAIN_OUTPUT_DIR}"
echo "  Corpus: ${TRAIN_CORPUS}"
echo ""

# Validate corpus exists
if [[ ! -f "${TRAIN_CORPUS}" ]]; then
    echo "ERROR: Training corpus not found: ${TRAIN_CORPUS}"
    echo "Please run preprocess.sbatch first"
    exit 1
fi

CORPUS_SIZE=$(wc -c < "${TRAIN_CORPUS}")
echo "Corpus size: ${CORPUS_SIZE} bytes"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "Starting training..."
echo ""

# Run training
python -m src.training.train \
    --corpus "${TRAIN_CORPUS}" \
    --model "${TRAIN_MODEL}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --output-dir "${TRAIN_OUTPUT_DIR}"

echo ""
echo "======================================================================"
echo "Training job completed successfully"
echo "Model saved to: ${TRAIN_OUTPUT_DIR}"
echo "Finished: $(date)"
echo "======================================================================"

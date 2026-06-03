#!/bin/bash
#SBATCH --account=e32706        ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu      ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00          ## Increase for gpt2-medium or larger models
#SBATCH --nodes=1               ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1              ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=40G               ## More headroom for model load, checkpoints, and Trainer
#SBATCH --job-name=text_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "======================================================================"
echo "Job: train GPT-2/Gemma 3 model"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Started: $(date)"
echo "======================================================================"

PROJECT_ROOT="$(cd ../.. && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

module load mamba/24.3.0

VENV_PATH="${GENAI_TEXT_VENV:-${HOME}/.venvs/genai-text}"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

python --version

TRAIN_MODEL="${TRAIN_MODEL:-gpt2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-models/nietzsche-bot}"
TRAIN_CORPUS="${TRAIN_CORPUS:-src/data/processed/training_corpus.txt}"
USE_LORA="${USE_LORA:-false}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"

echo ""
echo "Training configuration:"
echo "  Model: ${TRAIN_MODEL}"
echo "  Epochs: ${TRAIN_EPOCHS}"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  Output directory: ${TRAIN_OUTPUT_DIR}"
echo "  Corpus: ${TRAIN_CORPUS}"
echo "  Use LoRA: ${USE_LORA}"
echo "  4-bit quantization: ${LOAD_IN_4BIT}"
echo ""

if [[ ! -f "${TRAIN_CORPUS}" ]]; then
    echo "ERROR: Training corpus not found: ${TRAIN_CORPUS}"
    echo "Please run preprocess.sh first"
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

LORA_ARG=""
if [[ "${USE_LORA}" == "true" ]]; then
    LORA_ARG="--use-lora"
fi

QUANT_ARG=""
if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
    QUANT_ARG="--load-in-4bit"
fi

python "${PROJECT_ROOT}/src/training/train.py" \
    --corpus "${TRAIN_CORPUS}" \
    --model google/gemma-3-1b-pt \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --use-lora \
    --load-in-4bit

echo ""
echo "======================================================================"
echo "Training job completed successfully"
echo "Model saved to: ${TRAIN_OUTPUT_DIR}"
echo "Finished: $(date)"
echo "======================================================================"

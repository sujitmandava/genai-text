#!/bin/bash
#SBATCH --account=e32706        ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu      ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00          ## Gemma-3-1b LoRA+4bit fits on one GPU; bump up for larger models/corpora
#SBATCH --nodes=1               ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1              ## How many CPUs or processors do you need? (default value 1)
#SBATCH --cpus-per-task=4       ## Dataloader / tokenization workers
#SBATCH --mem=48G               ## Headroom for 4-bit load, bitsandbytes, checkpoints, and Trainer
#SBATCH --job-name=gemma_lora_train
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

TRAIN_MODEL="${TRAIN_MODEL:-google/gemma-3-1b-pt}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-models/gemma-nietzsche-lora}"
TRAIN_CORPUS="${TRAIN_CORPUS:-src/data/processed/training_corpus.txt}"
USE_LORA="${USE_LORA:-true}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
TRAIN_RESUME="${TRAIN_RESUME:-}"

echo ""
echo "Training configuration:"
echo "  Model: ${TRAIN_MODEL}"
echo "  Epochs: ${TRAIN_EPOCHS}"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  Output directory: ${TRAIN_OUTPUT_DIR}"
echo "  Corpus: ${TRAIN_CORPUS}"
echo "  Use LoRA: ${USE_LORA}"
echo "  4-bit quantization: ${LOAD_IN_4BIT}"
echo "  Resume from: ${TRAIN_RESUME:-<none>}"
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

RESUME_ARG=""
if [[ -n "${TRAIN_RESUME}" ]]; then
    RESUME_ARG="--resume ${TRAIN_RESUME}"
fi

python "${PROJECT_ROOT}/src/training/train.py" \
    --corpus "${TRAIN_CORPUS}" \
    --model "${TRAIN_MODEL}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    ${LORA_ARG} \
    ${QUANT_ARG} \
    ${RESUME_ARG}

echo ""
echo "======================================================================"
echo "Training job completed successfully"
echo "Model saved to: ${TRAIN_OUTPUT_DIR}"
echo "Finished: $(date)"
echo "======================================================================"

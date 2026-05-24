#!/bin/bash
#SBATCH --account=e32706        ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu      ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00          ## Increase for larger validation runs
#SBATCH --nodes=1               ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1              ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=16G               ## Headroom for model load and evaluation
#SBATCH --job-name=text_evaluate
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "======================================================================"
echo "Job: evaluate GPT-2 model"
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

EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-models/nietzsche-bot/final}"
EVAL_CORPUS="${EVAL_CORPUS:-src/data/processed/training_corpus.txt}"
EVAL_OUTPUT="${EVAL_OUTPUT:-evaluation_report.json}"

echo ""
echo "Evaluation configuration:"
echo "  Checkpoint: ${EVAL_CHECKPOINT}"
echo "  Corpus: ${EVAL_CORPUS}"
echo "  Output: ${EVAL_OUTPUT}"
echo ""

if [[ ! -d "${EVAL_CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint directory not found: ${EVAL_CHECKPOINT}"
    echo "Set EVAL_CHECKPOINT or run train.sh first"
    exit 1
fi

if [[ ! -f "${EVAL_CORPUS}" ]]; then
    echo "ERROR: Training corpus not found: ${EVAL_CORPUS}"
    echo "Please run preprocess.sh first"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

echo "Starting evaluation..."
echo ""

python "${PROJECT_ROOT}/src/training/evaluate.py" \
    --checkpoint "${EVAL_CHECKPOINT}" \
    --corpus "${EVAL_CORPUS}" \
    --output "${EVAL_OUTPUT}"

echo ""
echo "======================================================================"
echo "Evaluation job completed successfully"
echo "Report saved to: ${EVAL_OUTPUT}"
echo "Finished: $(date)"
echo "======================================================================"

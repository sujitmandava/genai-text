#!/usr/bin/env bash
#SBATCH --account=fkn5296
#SBATCH --job-name=text-preprocess
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/slurm/text-preprocess-%j.out
#SBATCH --error=logs/slurm/text-preprocess-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Preprocess raw texts for genai-text project
# CPU-only job that creates training corpus and extracts passages for RAG

set -euo pipefail

# Print job context
echo "======================================================================"
echo "Job: preprocess texts"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
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

# Validate raw data exists
RAW_DIR="src/data/raw"
if [[ ! -d "${RAW_DIR}" ]]; then
    echo "ERROR: Raw data directory not found: ${RAW_DIR}"
    exit 1
fi

TXT_COUNT=$(find "${RAW_DIR}" -name "*.txt" -type f | wc -l)
if [[ ${TXT_COUNT} -eq 0 ]]; then
    echo "ERROR: No .txt files found in ${RAW_DIR}"
    echo "Please run download.sbatch first"
    exit 1
fi

echo "Found ${TXT_COUNT} text files in ${RAW_DIR}"
echo ""
echo "Starting preprocessing..."
echo ""

# Preprocess texts using Python heredoc
python - <<PY
import sys
sys.path.insert(0, '${REPO_ROOT}')

from src.data.clean_for_training import create_training_corpus
from src.data.extract_passages import extract_all_passages

print("Step 1: Creating training corpus...")
create_training_corpus()

print("\nStep 2: Extracting passages for RAG...")
extract_all_passages()

print("\nPreprocessing complete.")
PY

echo ""
echo "======================================================================"
echo "Preprocess job completed successfully"
echo "Finished: $(date)"
echo "======================================================================"

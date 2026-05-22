#!/usr/bin/env bash
#SBATCH --account=fkn5296
#SBATCH --job-name=text-download
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/slurm/text-download-%j.out
#SBATCH --error=logs/slurm/text-download-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Download raw texts from Project Gutenberg for genai-text project
# This is a lightweight CPU-only job that downloads raw texts without preprocessing

set -euo pipefail

# Print job context
echo "======================================================================"
echo "Job: download raw texts"
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

# Parse FORCE_DOWNLOAD env var (default: false)
FORCE_ARG="False"
if [[ "${FORCE_DOWNLOAD:-0}" =~ ^(1|true|yes)$ ]]; then
    FORCE_ARG="True"
    echo "Force download enabled"
fi

echo ""
echo "Starting download..."
echo ""

# Download texts using Python heredoc
python - <<PY
import sys
sys.path.insert(0, '${REPO_ROOT}')

from src.data.download import download_all_texts

print("Calling download_all_texts(force=${FORCE_ARG})...")
download_all_texts(force=${FORCE_ARG})
print("Download complete.")
PY

echo ""
echo "======================================================================"
echo "Download job completed successfully"
echo "Finished: $(date)"
echo "======================================================================"

#!/bin/bash
#SBATCH --account=e32706        ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu      ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00          ## Project Gutenberg downloads are small, but network can be slow
#SBATCH --nodes=1               ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1              ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=8G                ## Headroom for dependency install and text downloads
#SBATCH --job-name=text_download
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "======================================================================"
echo "Job: download raw texts"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started: $(date)"
echo "======================================================================"

PROJECT_ROOT="$(cd ../ && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

mkdir -p logs/slurm

module load mamba/24.3.0

VENV_PATH="${GENAI_TEXT_VENV:-${HOME}/.venvs/genai-text}"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

python --version

DOWNLOAD_ARGS=()
if [[ "${FORCE_DOWNLOAD:-0}" =~ ^(1|true|yes)$ ]]; then
    DOWNLOAD_ARGS+=(--force)
    echo "Force download enabled"
fi

echo ""
echo "Starting download..."
echo ""

python "${PROJECT_ROOT}/src/data/download.py" "${DOWNLOAD_ARGS[@]}"

echo ""
echo "======================================================================"
echo "Download job completed successfully"
echo "Finished: $(date)"
echo "======================================================================"

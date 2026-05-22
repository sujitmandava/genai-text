#!/bin/bash
#SBATCH --account=e32706        ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu      ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00          ## Text preprocessing is small, but includes dependency install
#SBATCH --nodes=1               ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1              ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=8G                ## Headroom for dependency install and text processing
#SBATCH --job-name=text_preprocess
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "======================================================================"
echo "Job: preprocess texts"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started: $(date)"
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
echo "Working directory: ${REPO_ROOT}"

mkdir -p logs/slurm

module load mamba/24.3.0

VENV_PATH="${GENAI_TEXT_VENV:-${HOME}/.venvs/genai-text}"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

python --version

RAW_DIR="src/data/raw"
if [[ ! -d "${RAW_DIR}" ]]; then
    echo "ERROR: Raw data directory not found: ${RAW_DIR}"
    exit 1
fi

TXT_COUNT=$(python - <<PY
from pathlib import Path
print(len(list(Path("${RAW_DIR}").glob("*.txt"))))
PY
)
if [[ ${TXT_COUNT} -eq 0 ]]; then
    echo "ERROR: No .txt files found in ${RAW_DIR}"
    echo "Please run download.sh first"
    exit 1
fi

echo "Found ${TXT_COUNT} text files in ${RAW_DIR}"
echo ""
echo "Starting preprocessing..."
echo ""

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

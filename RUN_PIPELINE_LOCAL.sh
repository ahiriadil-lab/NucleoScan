#!/bin/bash
# RUN_PIPELINE_LOCAL.sh
#
# Run NucleoScan locally on all proteins in the dataset.
# Requires a CUDA-capable GPU and an activated Python environment.
#
# Usage:
#   bash RUN_PIPELINE_LOCAL.sh
#   bash RUN_PIPELINE_LOCAL.sh --database v3 --proteins Ubiquitin CI2
#   bash RUN_PIPELINE_LOCAL.sh --skip-advanced
#
# Logs are written to logs/dataset_run.log

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"

# Activate virtual environment if it exists
if [ -d "$PROJ/venv_esmfold" ]; then
    source "$PROJ/venv_esmfold/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: no virtual environment found. Ensure dependencies are installed."
fi

# Check GPU availability
python3 -c "import torch; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"not available\"}')"

echo ""
echo "============================================================"
echo " NucleoScan"
echo " $(date)"
echo "============================================================"
echo ""

python3 run_dataset.py "$@"

echo ""
echo "============================================================"
echo " Pipeline complete — $(date)"
echo " Results: $PROJ/results/"
echo "============================================================"

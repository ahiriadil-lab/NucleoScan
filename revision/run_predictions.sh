#!/bin/bash
# ESMFold predictions in the biochunan/esmfold-image container (fair-esm + OpenFold, torch 1.12.1).
cd "$(dirname "$0")"
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -v "$PWD":/work -v "$HOME/.cache/torch/hub/checkpoints":/tmp/.cache/torch/hub/checkpoints:ro \
  --entrypoint /miniconda/envs/py39-esmfold/bin/python biochunan/esmfold-image -u /work/predict_docker.py

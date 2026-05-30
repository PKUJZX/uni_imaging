#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

CONFIG="${1:-configs/train_frozen_encoder_voxel.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29502}" \
  train.py \
  --config "$CONFIG" \
  "$@"

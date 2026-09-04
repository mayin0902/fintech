#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_DATASET_ID:?Set TRAIN_DATASET_ID to a verified-only JSONL file}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export NUM_NPUS="${NUM_NPUS:-2}"
export FSDP_SIZE="${FSDP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-1}"
export MODEL_ID="${MODEL_ID:-ms://Qwen/Qwen3.5-2B}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen35-text2sql-lora}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# HCCL needs a routable host interface for its root-info handshake.
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NUM_NPUS}" \
  framework/twinkle_sft_qwen35_npu.py


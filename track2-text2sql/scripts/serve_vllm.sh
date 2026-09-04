#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"

vllm serve "$MODEL_ID" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192

# Hardware/model permitting, benchmark FP8 KV cache separately:
#   --kv-cache-dtype fp8
# Never cargo-cult these values: sweep concurrency, context distribution, TTFT,
# inter-token latency, throughput, and accuracy on the deployment GPU.


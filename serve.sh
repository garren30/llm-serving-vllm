#!/usr/bin/env bash
# Arranque de vLLM — Qwen 2.5 1.5B en la RTX 4050
set -e

# Esquiva FlashInfer (necesitaría nvcc); usa la ruta Triton
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager
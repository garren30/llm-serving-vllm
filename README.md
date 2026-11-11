# LLM Serving with vLLM — Latency & Throughput Benchmarks

Serving an open-weights LLM behind an OpenAI-compatible API with **vLLM**, and
measuring how it behaves under load on a single consumer GPU.

The goal of this project is not to serve the biggest model — it is to **measure
serving behavior rigorously** (latency percentiles, throughput, the effect of
continuous batching) and to understand *why* the numbers look the way they do.

---

## TL;DR — Key result

On the **same GPU and the same model**, raising request concurrency from 1 to 16
increased throughput **from 54 to 744 tokens/s (13.7×)** thanks to vLLM's
**continuous batching** — no hardware or model change.

| Concurrency | Throughput |
| --- | --- |
| 1 (sequential) | 54.4 tok/s |
| 4 | 158.9 tok/s |
| 8 | 377.4 tok/s |
| 16 | **743.9 tok/s** |

This is the core lesson of LLM inference: a single request leaves the GPU heavily
**underutilized** (inference is memory-bound — the GPU waits on memory with idle
compute). Continuous batching fills those gaps by processing many requests in the
same GPU step.

---

## Architecture

```
client / benchmark script
        │  HTTP (OpenAI-compatible: POST /v1/chat/completions)
        ▼
   vLLM server  ──►  RTX 4050 (6 GB, Ada Lovelace)
   (continuous batching, paged KV cache)
```

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (open weights, no gating; ~3 GB in VRAM)
- **Server:** vLLM 0.33.0, OpenAI-compatible endpoint on `localhost:8000`
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace), WSL2 / Ubuntu 24.04

---

## Results

### 1. Sequential latency (20 requests, one at a time)

Fixed prompt, `max_tokens=128`, `temperature=0.7`.

| Metric | Value |
| --- | --- |
| Mean latency | 2.19 s |
| p50 | 2.28 s |
| p95 | 2.40 s |
| p99 | 2.45 s |
| Throughput | 53.7 tok/s |

**Why percentiles, not just the average:** the average hides the slow tail. A
p99 of 2.45 s means 99% of requests finish faster than that. Here p50 ≈ p99, so
the server is **very consistent** — no long tail under single-request load.

### 2. Throughput vs concurrency (continuous batching)

32 requests per level, sent with increasing parallelism.

See the table in the TL;DR. Throughput nearly doubled again from concurrency 8 to
16 (377 → 744 tok/s), meaning the GPU **still had headroom** at 8 — the saturation
point (where the 6 GB KV cache fills up) is higher and is the next thing to measure.

---

## Design decisions & trade-offs

- **`--enforce-eager`:** disables `torch.compile` / CUDA graphs. On a 6 GB laptop
  GPU this **saves VRAM** and avoids extra build-time dependencies, at the cost of
  some per-token speed. A compile-on vs compile-off comparison is planned.
- **`--gpu-memory-utilization 0.80`:** a laptop GPU also drives the Windows
  display (~1 GB always in use), leaving ~4.9 GB free. 0.80 fits; 0.85 OOMs.
- **Small model on purpose:** Qwen 1.5B fits comfortably in 6 GB and keeps the
  focus on *serving methodology*, which transfers to any model size.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`:** avoids a runtime CUDA-toolkit (nvcc)
  dependency by using the Triton sampling path instead of FlashInfer.

---

## How to run

```bash
# 1. Start the server (one terminal)
./serve.sh

# 2. Benchmark (another terminal, once "Application startup complete" shows)
python bench_serve.py      # sequential latency / throughput
python bench_parallel.py   # throughput vs concurrency
```

Requires an NVIDIA GPU, `vllm`, and Python 3.12. See `serve.sh` for the exact
server flags.

---

## What's next

- Find the saturation point (concurrency 32 / 64) and the latency/throughput trade-off curve.
- Containerize the server with Docker.
- Deploy on a local Kubernetes cluster (kind/minikube).
- Quantization (AWQ) and a compile-on vs compile-off before/after (next project).

---

*Part of a hands-on path into MLOps / Inference Engineering. Measured on real
hardware, not copied from docs.*
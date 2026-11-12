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
16 (377 → 744 tok/s), meaning the GPU still had headroom — so the next step was to
find where it actually saturates.

### 3. Saturation point (throughput + latency vs concurrency)

Pushing concurrency further, measuring throughput **and** latency together:

| Concurrency | Throughput | p50 | p99 |
| --- | --- | --- | --- |
| 32 | 1318 tok/s | 2.78 s | 2.93 s |
| 64 | 2212 tok/s | 3.29 s | 3.34 s |
| 128 | 2250 tok/s | 3.21 s | 3.27 s |
| 256 | 2298 tok/s | 3.28 s | 3.31 s |

**The GPU saturates around concurrency 64.** From 32→64 throughput nearly doubles,
but from 64→256 it moves only +4% (2212 → 2298 tok/s) despite 4× more concurrency —
and p99 latency stays flat (~3.3 s) instead of exploding. vLLM absorbs the overload
by **queueing requests in order** rather than degrading each one.

**Capacity takeaway:** this configuration comfortably serves up to ~64 concurrent
requests, with a throughput ceiling of ~2250 tok/s. Beyond that, adding concurrency
buys no extra useful work — exactly the kind of number you size a real system on.

### 4. Output length drives latency (~18 ms/token, constant)

Same input prompt, varying `max_tokens` on the response (`ignore_eos=True` forces
an exact length). Input tokens stay fixed at 35.

| Output tokens | Time | ms / token |
| --- | --- | --- |
| 32 | 0.59 s | 18.5 ms |
| 64 | 1.18 s | 18.4 ms |
| 128 | 2.35 s | 18.4 ms |
| 256 | 4.70 s | 18.4 ms |
| 512 | 9.45 s | 18.5 ms |

Latency grows **linearly** with output length (512 tokens = 16× the time of 32),
and the **per-token cost stays flat at ~18.4 ms** regardless of length. This is the
autoregressive decode loop: each output token is one sequential step of fixed cost.

**Cost-model takeaway:** request latency is driven by **output length**, not input
length — the generated tokens dominate. This is why `max_tokens` is the main knob
for controlling latency and cost per request.

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
- **Docker with the official vLLM image:** the native setup needed five system
  dependencies solved by hand (driver, gcc, Python headers, nvcc, memory flags).
  The container ships all of that pre-built — the host only has to pass in the GPU
  (`--gpus all`) and map the port. This is the whole point of containerizing a
  GPU service: a complex environment packaged once, reproduced anywhere.

---

## How to run

### Option A — Docker (recommended, reproducible)

The server is packaged with the official vLLM image and a `docker-compose.yml`
that wires up the GPU, port, and flags. One command, same result on any machine
with Docker + an NVIDIA GPU:

```bash
docker compose up
```

GPU access inside the container is declared in `docker-compose.yml`
(`deploy.resources.reservations.devices`, equivalent to `--gpus all`). No manual
dependency setup — the image ships the full CUDA/vLLM stack.

### Option B — Native (local venv)

```bash
# 1. Start the server (one terminal)
./serve.sh

# 2. Benchmark (another terminal, once "Application startup complete" shows)
python bench_serve.py        # sequential latency / throughput
python bench_parallel.py     # throughput vs concurrency
python bench_saturation.py   # throughput + latency vs concurrency (finds saturation)
python bench_outlen.py       # latency vs output length (max_tokens)
```

Requires an NVIDIA GPU, `vllm`, and Python 3.12. See `serve.sh` for the exact
server flags.

---

## What's next

- Deploy on a local Kubernetes cluster (kind/minikube).
- Quantization (AWQ) and a compile-on vs compile-off before/after (next project).

---

*Part of a hands-on path into MLOps / Inference Engineering. Measured on real
hardware, not copied from docs.*
# Future Stage Roadmap

> Provenance: LLM-generated draft. Review and edit by hand before treating this as project authority.

This roadmap captures the later stages for a miniature Rust + Triton LLM serving engine. The intent is sequencing: each stage should produce a small artifact that makes the next stage easier.

## Stage 0: Systems Prerequisites

Goal: become comfortable enough with systems concepts that later GPU, runtime, and scheduler work has a foundation.

Learn:

- Memory layout and cache locality
- Stack versus heap
- Pointers and ownership concepts
- Threads, mutexes, atomics, and contention
- SIMD at a conceptual level
- CPU versus GPU execution models

Milestones:

- Write notes on memory hierarchy and locality.
- Implement small CPU benchmarks that show cache effects.
- Compare single-threaded and multi-threaded throughput for a simple workload.

Exit criteria:

- You can explain why memory access patterns often dominate raw arithmetic cost.
- You can identify when shared mutable state creates synchronization risk.

## Stage 1: Transformer Inference Internals

Goal: understand the model execution path well enough that serving bottlenecks make sense.

Detailed plan:

- [Stage 1 Milestones: Transformer Inference Internals](./stage-1-transformer-inference.md)

Core outputs:

- Minimal PyTorch attention implementation
- Traced autoregressive generation loop
- Tokenization observations
- KV cache measurements
- Small FastAPI inference playground

Exit criteria:

- You can explain attention, prefill, decode, KV cache, batching, and why decode is often memory-bound.

## Stage 2: PyTorch Runtime And Graph Basics

Goal: understand enough PyTorch internals to inspect model execution and prepare for kernel/runtime work.

Learn:

- Tensor storage, strides, views, and contiguity
- Dtypes and device placement
- Autograd basics, even though this project focuses on inference
- `torch.no_grad()` and inference mode
- TorchScript, FX graphs, and `torch.compile` at a high level
- Graph capture limitations caused by dynamic generation and KV cache growth

Milestones:

- Instrument a small transformer forward pass with tensor shape, stride, dtype, and device logs.
- Use FX or `torch.compile` on a simple model and inspect what does and does not get captured.
- Compare eager execution and compiled execution for stable-shape toy workloads.

Exit criteria:

- You can explain why dynamic shapes and cache growth complicate graph capture.
- You can distinguish a view from a copy and explain why contiguity matters for kernels.

## Stage 3: GPU Architecture

Goal: understand the hardware model behind Triton and CUDA performance.

Learn:

- Streaming multiprocessors
- Warps and SIMT execution
- Occupancy
- Global memory, shared memory, registers, and caches
- Coalesced memory access
- Tensor cores
- Synchronization and memory movement

Milestones:

- Write notes mapping GPU memory hierarchy to LLM inference bottlenecks.
- Profile a simple PyTorch operation and identify GPU time versus CPU overhead.
- Explain prefill and decode in terms of compute intensity and memory bandwidth.

Exit criteria:

- You can reason about why a kernel might be memory-bound, compute-bound, or launch-overhead-bound.

## Stage 4: Triton Kernel Foundations

Goal: learn Triton through small kernels before touching production-grade attention.

Learn:

- Program IDs
- Block tiling
- Vectorized loads and stores
- Masks for boundary handling
- Reductions
- Matrix multiplication tiling
- Numerically stable softmax

Milestones:

- Implement vector add in Triton.
- Implement tiled matrix multiplication and compare against PyTorch.
- Implement row-wise softmax and test numerical stability.
- Build a toy attention kernel or FlashAttention-style educational prototype.

Exit criteria:

- You can explain how tiling improves data reuse.
- You can benchmark a Triton kernel and compare correctness against PyTorch.

## Stage 5: Profiling And Performance Engineering

Goal: learn to answer why something is slow with measurements instead of guesses.

Learn:

- Nsight Systems
- Nsight Compute
- Kernel launch overhead
- Occupancy
- Warp stalls
- Memory throughput
- Tensor core utilization
- Roofline-style reasoning

Milestones:

- Profile the Stage 4 matrix multiplication kernel.
- Profile softmax or attention and identify the dominant bottleneck.
- Create a benchmark report with latency, throughput, bandwidth estimate, and correctness checks.

Exit criteria:

- You can connect profiler metrics to a concrete optimization hypothesis.
- You can distinguish useful performance wins from measurement noise.

## Stage 6: LLM Serving Systems

Goal: move from single-request inference to multi-request serving behavior.

Learn:

- Request queues
- Static batching
- Continuous batching
- Scheduler policies
- Latency versus throughput tradeoffs
- KV cache allocation, paging, fragmentation, and eviction
- Quantization concepts: FP16, BF16, INT8, FP8

Milestones:

- Build a Python serving prototype with request queuing and token streaming.
- Add naive batching and measure time to first token, inter-token latency, and throughput.
- Implement a simple KV cache accounting model.
- Read vLLM/PagedAttention closely and summarize its memory-management design.

Exit criteria:

- You can explain how continuous batching and KV cache management interact.
- You can describe why serving is a runtime systems problem, not just a model call.

## Stage 7: Rust Runtime Layer

Goal: learn Rust for the serving control plane: networking, scheduling, streaming, and metrics.

Learn:

- Ownership and borrowing
- Lifetimes conceptually
- `Arc`, `Mutex`, and channels
- Async Rust
- Tokio
- Axum
- Server-sent events or WebSocket streaming
- Metrics and tracing

Milestones:

- Build a small Axum HTTP service.
- Add streaming token responses with a mock backend.
- Implement a request queue and scheduler loop.
- Add metrics for queue depth, request latency, and generated tokens.

Exit criteria:

- You can design a Rust scheduler without fighting the borrow checker on every state transition.
- You can explain where shared mutable state exists and how it is protected.

## Stage 8: Integrated Mini Serving Engine

Goal: combine a Rust serving layer with a Python/Triton model execution layer.

Target architecture:

- Rust HTTP API
- Rust request queue
- Rust batching scheduler
- Streaming responses to clients
- Python model execution backend
- Triton kernels for selected operations
- Metrics and benchmark harness

Milestones:

- Define the Rust-to-Python backend boundary.
- Implement request admission, cancellation, and streaming.
- Add a batching scheduler that can group compatible decode steps.
- Integrate model execution and expose basic metrics.
- Benchmark single-request, static batching, and continuous-style batching behavior.

Exit criteria:

- You can run an end-to-end local inference server and explain every major subsystem.
- You have benchmark results tied to architecture choices.
- You can identify the next highest-impact optimization from evidence.

## Stretch Goals

Only start these after the integrated engine works:

- Paged KV cache allocator
- Prefix cache reuse
- Speculative decoding
- CUDA graph capture
- Tensor parallelism
- Pipeline parallelism
- Quantized KV cache
- OpenAI-compatible API surface

## Suggested Six-Month Sequence

Month 1:

- Stage 1 transformer inference basics
- PyTorch tensors
- Simple inference loop

Month 2:

- Stage 2 PyTorch runtime basics
- Stage 3 GPU architecture
- First Triton kernels

Month 3:

- Stage 4 matrix multiplication, softmax, and attention prototypes
- Stage 5 profiling with Nsight

Month 4:

- Stage 6 mini inference server
- Batching experiments
- KV cache accounting

Month 5:

- Stage 7 Rust async, Tokio, Axum, streaming, and scheduler prototype

Month 6:

- Stage 8 Rust scheduler plus Python backend integration
- Benchmarks and final project report

## Engineering Bar

The project should be judged by concrete systems claims, not technology name-dropping.

Good outcomes look like:

- Reduced decode latency by changing batching behavior.
- Measured KV cache memory growth across concurrent requests.
- Explained a kernel bottleneck using profiler evidence.
- Improved throughput while documenting the latency tradeoff.

Avoid claims like:

- Used Rust.
- Used CUDA or Triton.
- Built an LLM server.

Those are implementation details. The valuable part is understanding and measuring the system behavior.

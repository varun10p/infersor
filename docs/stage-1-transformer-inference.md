# Stage 1 Milestones: Transformer Inference Internals

> Provenance: LLM-generated draft. Review and edit by hand before treating this as project authority.

Stage 1 goal: understand transformer inference deeply enough that LLM serving systems make sense. The focus is inference, not training or model research.

By the end of this stage, you should be able to explain the full path from text prompt to generated tokens, including attention, tokenization, prefill, decode, KV cache growth, batching pressure, and why modern serving systems spend so much effort on memory management.

## Milestone 1: Transformer And GPT Basics

### Learn

- Token embeddings and positional information
- Self-attention and causal masking
- Multi-head attention
- Residual connections and layer normalization
- Decoder-only GPT-style generation

### Primary Material

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Deliverables

- Write `docs/notes/transformer-basics.md` explaining a decoder-only transformer in your own words.
- Include the tensor shapes for `tokens`, `embeddings`, `Q`, `K`, `V`, attention scores, attention probabilities, and final logits.
- Explain what causal masking prevents during generation.

### Exit Criteria

You can draw the forward pass for one transformer block and explain why the model can only predict the next token from previous tokens.

## Milestone 2: Attention Math And Tensor Shapes

### Learn

- Scaled dot-product attention
- Why attention uses `QK^T`
- Why the scale factor `sqrt(d_k)` matters
- How softmax converts scores into weights
- Why attention cost grows with sequence length

### Primary Material

- [Attention Is All You Need, section 3.2](https://arxiv.org/abs/1706.03762)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [PyTorch Tensor Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)

### Deliverables

- Implement a tiny attention function in PyTorch using raw tensor operations.
- Print tensor shapes at each step.
- Add a causal mask and verify that token `i` cannot attend to token `j > i`.

### Exit Criteria

You can explain this equation using concrete tensor shapes:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

## Milestone 3: Autoregressive Generation

### Learn

- Next-token prediction
- Prompt processing
- Token-by-token decode loop
- Sampling basics: greedy, temperature, top-k, top-p
- Why generation repeatedly calls the model

### Primary Material

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Let's Build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

### Deliverables

- Run a small GPT-style model locally.
- Trace the generation loop and document where logits become the next token.
- Add logging for prompt length, generated token count, and per-token latency.

### Exit Criteria

You can explain why generating 100 tokens is not one model call, but roughly 100 decode steps after prefill.

## Milestone 4: Tokenization

### Learn

- Tokens versus characters and words
- BPE-style subword tokenization
- Why token count affects latency and memory
- Context window limits

### Primary Material

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Hugging Face Tokenizers Course](https://huggingface.co/learn/llm-course/chapter6/1)
- [Let's Build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

### Deliverables

- Compare token counts for short code, prose, JSON, and whitespace-heavy input.
- Document examples where character length and token length differ significantly.
- Explain how token count affects prefill and KV cache size.

### Exit Criteria

You can estimate why two prompts of similar character length may have different inference costs.

## Milestone 5: KV Cache And Prefill Versus Decode

### Learn

- What keys and values are cached
- Why recomputing past keys and values is wasteful
- Why KV cache grows with sequence length
- Prefill as the prompt-processing phase
- Decode as the one-token-at-a-time generation phase
- Why prefill is usually compute-heavy and decode is often memory-bandwidth-heavy

### Primary Material

- [Hugging Face KV Cache Explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Hugging Face KV Cache Strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### Deliverables

- Run generation with and without KV cache using a small Hugging Face model.
- Measure tokens per second and memory use for increasing prompt lengths.
- Write `docs/notes/kv-cache.md` with a simple size formula for KV cache memory.

### Exit Criteria

You can explain why decode latency depends heavily on reading model weights and KV cache, and why serving many concurrent requests becomes a memory-management problem.

## Milestone 6: PyTorch Tensor Inspection For Inference

### Learn

- Tensor shape, dtype, and device
- Strides
- Views versus copies
- `transpose`, `view`, `reshape`, and `contiguous`
- Broadcasting
- `torch.no_grad()` and inference mode

### Primary Material

- [PyTorch Tensor Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)
- [PyTorch Introduction](https://docs.pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)
- [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)

### Deliverables

- Add shape and stride logging around an attention implementation.
- Create examples showing when `transpose` produces a non-contiguous tensor.
- Document where contiguous layout matters for later kernel work.

### Exit Criteria

You can inspect an attention tensor and explain its shape, stride, dtype, device, and whether it is contiguous.

## Milestone 7: Why Inference Serving Is Hard

### Learn

- Static batching versus dynamic batching
- Continuous batching
- Request arrival variability
- Time to first token versus throughput
- KV cache fragmentation
- Why serving systems separate scheduling from model execution

### Primary Material

- [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)
- [Hugging Face Continuous Batching Docs](https://huggingface.co/docs/transformers/main/continuous_batching)
- [vLLM PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Hugging Face Optimizing Inference Guide](https://huggingface.co/docs/transformers/main/llm_optims)

### Deliverables

- Build a simple FastAPI endpoint that accepts prompts and streams generated tokens from a small local model.
- Add a naive queue and batch together requests that arrive within a short time window.
- Record latency, time to first token, and throughput for one request versus multiple concurrent requests.

### Exit Criteria

You can explain the tradeoff between latency and throughput, and why naive batching wastes GPU time for variable-length generations.

## Milestone 8: FlashAttention Conceptual Preview

### Learn

- Why standard attention moves too much data through HBM
- IO awareness
- Tiling attention computation
- Online softmax intuition
- Why kernel-level work matters later, without implementing kernels yet

### Primary Material

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Hugging Face Optimizing Inference Guide](https://huggingface.co/docs/transformers/main/llm_optims)

### Deliverables

- Write a one-page explanation of FlashAttention at the systems level.
- Compare standard attention and FlashAttention in terms of memory movement, not code.

### Exit Criteria

You can explain the sentence: attention is often bottlenecked by memory movement, not just floating-point operations.

## Final Stage 1 Project

Build a minimal inference playground:

- Loads a small Hugging Face causal language model.
- Accepts prompts through an HTTP API.
- Streams generated tokens.
- Logs prompt tokens, generated tokens, prefill time, decode time, tokens per second, and memory usage where available.
- Supports a simple batching experiment, even if it is intentionally naive.

Recommended stack:

- Python
- PyTorch
- Hugging Face Transformers
- FastAPI

### Final Report

Create `docs/stage-1-report.md` with:

- Architecture of your mini inference loop
- Tensor shape walkthrough
- Tokenization observations
- KV cache measurements
- Prefill versus decode timing
- Batching experiment results
- What remains unclear before Stage 2

## Suggested Four-Week Schedule

### Week 1: Transformer And Attention Basics

Complete milestones 1 and 2. The output should be notes and a working attention function with shape logging.

### Week 2: Generation And Tokenization

Complete milestones 3 and 4. The output should be a traced generation loop and tokenization observations.

### Week 3: KV Cache And PyTorch Inspection

Complete milestones 5 and 6. The output should be KV cache measurements and tensor layout notes.

### Week 4: Serving Concepts And Final Project

Complete milestones 7 and 8. The output should be the minimal inference playground and final Stage 1 report.

## Most Important Resources

If time is limited, prioritize these:

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
2. [Let's Build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)
3. [nanoGPT](https://github.com/karpathy/nanoGPT)
4. [Hugging Face KV Cache Explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
5. [vLLM PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)

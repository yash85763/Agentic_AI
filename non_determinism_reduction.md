I’ll access that page for you and explain the study in detail.​​​​​​​​​​​​​​​​

This is a fascinating and technically deep study on achieving deterministic results from Large Language Models. Let me break down the key findings and methodology:

## The Core Problem

The researchers  tackle a fundamental issue: even when using temperature=0 (which should make sampling deterministic), LLM APIs and inference engines still produce different results for identical inputs. This affects reproducibility, which is crucial for scientific research.

## Debunking the Common Theory

The widely accepted explanation has been the “concurrency + floating point” hypothesis - that nondeterminism comes from GPU parallel operations combined with floating-point arithmetic’s non-associative properties . However, the authors demonstrate this explanation is incomplete.

They show that even simple matrix multiplication on GPUs with the same inputs produces bitwise identical results repeatedly, despite using floating-point numbers and GPU concurrency .

## The Real Culprit: Batch Invariance

The study reveals the true cause: **lack of batch invariance**. Here’s what this means:

When you make a request to an LLM server, the batch size your request gets processed with depends on the current server load, which is nondeterministic from your perspective. Since the kernels aren’t “batch-invariant,” different batch sizes can produce different numerical results for the same input .

For example, they demonstrate that matrix multiplication results differ when computed as:

- Single element: `torch.mm(a[:1], b)`
- Full batch then slice: `torch.mm(a, b)[:1]`

## Technical Deep Dive

The researchers analyze three key operations that must be made batch-invariant:

### 1. RMSNorm

The challenge is ensuring reduction order stays fixed regardless of batch size. When batch size decreases, typical implementations change reduction strategies to maintain performance, breaking batch invariance .

### 2. Matrix Multiplication

Matmuls face similar issues with “Split-K” strategies when batch dimensions become too small. Different tensor core instructions may be used for different batch sizes, affecting results .

### 3. Attention (Most Complex)

Attention is the most challenging because it reduces over both feature and sequence dimensions, and must handle various inference optimizations like chunked prefill and prefix caching .

## Implementation and Results

The team provides a working implementation using vLLM with FlexAttention backend. Their experimental results are striking:

Testing with Qwen-3-235B-A22B-Instruct, they generated 1000 completions at temperature=0. Without their fixes, they got 80 unique completions. With batch-invariant kernels, all 1000 completions were identical .

Interestingly, the divergence didn’t happen immediately - all completions were identical for the first 102 tokens, with differences appearing at token 103 .

## Performance Impact

While their unoptimized implementation showed about 2x slowdown (26 seconds vs 55 seconds for 1000 sequences), optimizations reduced this to 1.6x slowdown (42 seconds) .

## Applications: True On-Policy RL

One of the most significant applications is enabling true on-policy reinforcement learning. Currently, numerical differences between training and inference inadvertently turn on-policy RL into off-policy RL. With deterministic inference, they can achieve bitwise identical results between sampling and training .

Their experiments show that without off-policy correction, reward collapses during training, but with true on-policy RL (0 KL divergence between sampler and trainer), training proceeds smoothly .

## Broader Implications

This work challenges the ML community’s tendency to accept nondeterminism as inevitable in probabilistic systems. The authors argue against “defeatism” and demonstrate that with proper understanding, we can eliminate these sources of nondeterminism .

The study provides both theoretical insights and practical tools (available on GitHub) for achieving truly reproducible LLM inference, which is crucial for scientific rigor in AI research.
# AI Engineering
## A Systems-First Guide to LLM Mathematics, Architectures, Hardware, Training, Inference, Quantization, and Reinforcement Learning

> **The one-line thesis:** Modern AI engineering is the art of turning statistical learning into useful computation under real constraints: data, numerical precision, memory bandwidth, latency, cost, power, safety, and evaluation.
>
> A useful systems equation is:
>
> ```text
> Capability = architecture × data × optimization × post-training × inference system × hardware × evaluation
> ```
>
> A model is not merely a stack of matrices. It is a stack of matrices whose data must be moved through a physical machine, whose behavior must be trained by an objective, and whose output must be served under a latency-and-cost budget.
>
> **Relationship to the Agentic Engineering curriculum:** Agentic engineering sits *above* this guide. A harness decides what an agent should see and do; this guide explains the model, inference stack, and hardware that make those decisions possible. The interface is not a sharp wall: agentic workloads change inference patterns through long contexts, prefix caching, tool-call bursts, and long reasoning traces.

---

# Table of Contents

- [Part 0 — The map: what an AI engineer actually engineers](#part-0--the-map-what-an-ai-engineer-actually-engineers)
- [Part I — Mathematics you need before models](#part-i--mathematics-you-need-before-models)
- [Part II — From text to GPT-style language models](#part-ii--from-text-to-gpt-style-language-models)
- [Part III — Architecture families beyond the dense Transformer](#part-iii--architecture-families-beyond-the-dense-transformer)
- [Part IV — Training foundation models](#part-iv--training-foundation-models)
- [Part IV-A — Multimodality, retrieval, distillation, and evaluation](#part-iv-a--multimodality-retrieval-distillation-and-evaluation)
- [Part V — The physical substrate: CPU, GPU, TPU, LPU, memory, and networks](#part-v--the-physical-substrate-cpu-gpu-tpu-lpu-memory-and-networks)
- [Part VI — Inference and serving systems](#part-vi--inference-and-serving-systems)
- [Part VII — Quantization, compression, and efficient adaptation](#part-vii--quantization-compression-and-efficient-adaptation)
- [Part VIII — Reinforcement learning and post-training](#part-viii--reinforcement-learning-and-post-training)
- [Part IX — Current research frontier, 2025–June 2026](#part-ix--current-research-frontier-2025june-2026)
- [Part X — Engineering projects and capability milestones](#part-x--engineering-projects-and-capability-milestones)
- [Part XI — A 24-week learning path](#part-xi--a-24-week-learning-path)
- [Part XII — Canonical references and live sources](#part-xii--canonical-references-and-live-sources)

---

# Part 0 — The map: what an AI engineer actually engineers

## 0.1 The three loops

AI engineering looks large because it is really three linked loops.

```text
                 ┌────────────────────────────────────────┐
                 │             Learning loop              │
                 │ data → objective → gradients → weights │
                 └───────────────────┬────────────────────┘
                                     │ trained parameters
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Execution loop                                 │
│ tokens → model forward pass → probabilities → decode → tokens               │
└───────────────────┬─────────────────────────────────────────────────────────┘
                    │ requests, models, caches, telemetry
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Deployment loop                                │
│ hardware → kernels → batching → routing → observability → cost/SLO control  │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Learning loop**: train parameters so the model assigns useful probability distributions to sequences.
2. **Execution loop**: turn a prompt into generated tokens by repeatedly evaluating the model.
3. **Deployment loop**: serve many execution loops reliably, cheaply, and safely on actual hardware.

Reinforcement learning and post-training live between the first two loops: they adjust learned behavior using preference, reward, tool, or verifier feedback. Agentic engineering lives above all three: it builds systems that decide *when* a model should execute, what it should see, which tools it may use, and how success is verified.

## 0.2 The five questions that explain most engineering decisions

Whenever you encounter a new paper, model, hardware announcement, or optimization, ask:

1. **What computation is expensive?** Matrix multiplication, attention, routing, sampling, communication, or memory movement?
2. **Where is the state?** Weights, activations, optimizer states, the KV cache, an SSM state, a prefix cache, or a distributed parameter shard?
3. **What bottleneck moved?** Compute, HBM bandwidth, memory capacity, network bandwidth, scheduler overhead, or verifier quality?
4. **What trade-off was made?** Accuracy, latency, throughput, context length, cost, determinism, energy, or engineering complexity?
5. **What must be measured to prove value?** Quality, TTFT, TPOT, goodput, cost/request, energy/token, peak memory, or a task-specific evaluator?

This is the antidote to acronym fog. An AI paper usually introduces a new way to redistribute one of these costs.

## 0.3 A glossary of the quantities that keep reappearing

| Quantity | Meaning | Why an engineer cares |
|---|---|---|
| Parameter | A learned scalar in the model | Determines model storage and training-state size |
| Activation | Intermediate value produced during a forward pass | Dominates training memory for long sequences |
| FLOP | Floating-point operation | A compute-cost unit, but not a complete performance metric |
| HBM | High-bandwidth memory attached to an accelerator | Often the inference bottleneck |
| SRAM / cache | Small, fast memory close to compute | Kernels try to reuse data here before returning to HBM |
| Arithmetic intensity | FLOPs performed per byte moved | Helps predict whether work is compute- or bandwidth-bound |
| KV cache | Stored attention keys and values from prior tokens | Enables autoregressive decoding, but consumes memory as context grows |
| TTFT | Time to first token | Main interactive latency metric for prompt processing |
| TPOT | Time per output token | Main streaming-latency metric for generation |
| Throughput | Work completed per unit time | Usually tokens/second or requests/second |
| Goodput | Throughput that satisfies a latency target | Better production metric than raw throughput |

---

# Part I — Mathematics you need before models

> **Learning philosophy:** Do not postpone all implementation until after “finishing the math.” Learn each mathematical idea just before you need it, then make it concrete in code. A vector becomes less mysterious after you use it as a token embedding; the chain rule becomes less mysterious after you see it move an error signal through attention.

## 1.1 The minimum viable mathematics map

```text
Linear algebra ─────► embeddings, projections, attention, tensor parallelism
      │
      ├──► Calculus / autodiff ─────► backpropagation, optimization, RL gradients
      │
      ├──► Probability / information theory ─────► language modeling, sampling, loss, RL
      │
      ├──► Numerical computing ─────► BF16/FP8/FP4, stability, quantization
      │
      └──► Performance math ─────► roofline model, bandwidth, kernels, serving
```

You do **not** need every theorem from an undergraduate mathematics degree before building models. You do need fluency with a compact set of ideas.

## 1.2 Linear algebra: the language of neural networks

### What to learn

1. Scalars, vectors, matrices, and higher-order tensors.
2. Shapes and dimension checking.
3. Matrix-vector and matrix-matrix multiplication.
4. Dot products, cosine similarity, norms, distances, and projections.
5. Linear maps and changes of basis.
6. Rank, low-rank approximations, eigendecomposition, and singular value decomposition.
7. Block matrices and sharding intuition.

### The essential intuition

A matrix is not merely a rectangular collection of numbers. It represents a learned transformation of a geometric space.

If

\[
x \in \mathbb{R}^{d_{in}}, \quad W \in \mathbb{R}^{d_{in} \times d_{out}}, \quad y=xW \in \mathbb{R}^{d_{out}},
\]

then the model has taken one feature vector, mixed its coordinates according to the learned matrix, and represented it in a new feature space. In a Transformer, the same hidden state is projected into multiple learned views: queries, keys, values, gate activations, expert-router scores, and vocabulary logits.

### Why SVD matters later

Low-rank decomposition says that a large matrix can sometimes be approximated by a small number of important directions:

\[
W \approx U_r \Sigma_r V_r^T.
\]

This becomes practical in **LoRA**, where rather than training a full update \(\Delta W\), you train a low-rank approximation:

\[
W' = W + BA,
\]

with \(B \in \mathbb{R}^{d_{out}\times r}\), \(A \in \mathbb{R}^{r\times d_{in}}\), and \(r\ll \min(d_{in}, d_{out})\).

### Best resources

- **Book:** *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong. https://mml-book.github.io/
- **Visual intuition:** 3Blue1Brown, *Essence of Linear Algebra*. https://www.3blue1brown.com/topics/linear-algebra
- **Proof-oriented companion:** *Linear Algebra Done Right* — Sheldon Axler.
- **Practice:** `torch.matmul`, `einsum`, and writing shapes above every operation.

## 1.3 Calculus and automatic differentiation

### What to learn

- Derivatives and partial derivatives.
- Gradient vectors.
- The chain rule.
- Jacobians and vector-Jacobian products.
- Computational graphs.
- Gradient descent and why learning rates matter.
- Gradient clipping and exploding/vanishing gradients.

### The central idea

The gradient

\[
\nabla_\theta \mathcal{L}(\theta)
\]

points toward the local direction of greatest increase in loss. Gradient descent moves in the opposite direction:

\[
\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}.
\]

Deep learning does not manually construct a gigantic symbolic derivative. It builds a computation graph during the forward pass and applies the chain rule backward through that graph. Modern frameworks calculate **vector-Jacobian products** efficiently, which is why backpropagation scales even when a full Jacobian would be impossibly large.

### A chain-rule picture

```text
input x → linear layer → activation → loss
              │             │         │
              ▼             ▼         ▼
          parameters W   hidden h     scalar L

Backward pass:
∂L/∂W = ∂L/∂h × ∂h/∂W
```

### Best resources

- **Book:** *Understanding Deep Learning* — Simon J. D. Prince.
- **Interactive:** Andrej Karpathy, *micrograd*. https://github.com/karpathy/micrograd
- **Practical:** PyTorch Autograd mechanics. https://pytorch.org/docs/stable/notes/autograd.html

## 1.4 Probability, likelihood, and information theory

Language models produce distributions, not facts. Given previous tokens \(x_{<t}\), a decoder predicts:

\[
p_\theta(x_t\mid x_{<t}).
\]

For a sequence \(x_1,\dots,x_T\), the autoregressive factorization is:

\[
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t\mid x_{<t}).
\]

### Cross-entropy / negative log-likelihood

With the correct next token \(y_t\), pretraining minimizes:

\[
\mathcal{L}_{NLL} = -\sum_{t=1}^{T}\log p_\theta(y_t\mid x_{<t}).
\]

This is the same as maximizing likelihood. It punishes a model when it assigns low probability to the observed next token.

### Softmax

Given logits \(z\in\mathbb{R}^{|V|}\), softmax converts unnormalized scores to probabilities:

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

The model’s output layer contains one logit for every vocabulary token. Sampling and decoding happen *after* this distribution is created.

### Entropy and KL divergence

- **Entropy** measures uncertainty in a distribution.
- **KL divergence** measures how one distribution differs from another.
- In RLHF-style optimization, a KL penalty commonly keeps a policy from moving too far from a reference model.

\[
D_{KL}(\pi_\theta\|\pi_{ref}) = \sum_a \pi_\theta(a)\log\frac{\pi_\theta(a)}{\pi_{ref}(a)}.
\]

### Best resources

- *Information Theory, Inference, and Learning Algorithms* — David MacKay.
- *Probabilistic Machine Learning: An Introduction* — Kevin Murphy.
- Blitzstein and Hwang’s probability text for foundations.

## 1.5 Optimization: why AdamW, schedules, and clipping exist

Pure SGD uses the current mini-batch gradient. In practice, LLM training often uses **AdamW**, which maintains exponential moving averages of gradients and squared gradients, then applies decoupled weight decay.

The high-level behavior is:

```text
noisy gradient
  → smooth estimates of direction and scale
  → normalized adaptive update
  → learning-rate schedule determines step size over time
```

Important concepts:

- **Warmup:** begin with a small learning rate to avoid unstable early updates.
- **Cosine decay / linear decay:** gradually reduce step size after broad learning begins.
- **Weight decay:** encourages smaller weights but is not the same as L2 regularization in every optimizer implementation.
- **Gradient clipping:** caps extreme updates before they destabilize training.
- **Batch size:** changes both hardware utilization and the noise properties of optimization.

### Best resources

- *Deep Learning* — Goodfellow, Bengio, Courville, chapters on optimization.
- *Numerical Optimization* — Nocedal and Wright, for a deeper treatment.
- PyTorch optimizer docs and a small AdamW-from-scratch exercise.

## 1.6 Numerical computing: why BF16, FP8, and FP4 are not footnotes

A floating-point format trades range and precision. Broadly:

- **FP32**: large range and precision, expensive in memory/bandwidth.
- **FP16**: compact, but limited exponent range can cause overflow/underflow.
- **BF16**: FP32-like exponent range with fewer mantissa bits; widely useful in training.
- **FP8 / FP4**: lower precision formats used when hardware and scaling strategies can preserve quality.
- **INT8 / INT4**: integer formats common in quantized inference.

A low-bit format is not inherently “better.” It is useful only when:

1. the model remains accurate enough;
2. kernels exploit the format efficiently;
3. data movement is actually reduced; and
4. conversions, scales, and dequantization do not erase the savings.

This will reappear in Part VII.

## 1.7 Performance mathematics: FLOPs are not enough

The **roofline model** gives a simple but powerful performance bound:

\[
\text{attainable performance} \leq \min(\text{peak FLOPs/s},\; \text{memory bandwidth}\times\text{arithmetic intensity}).
\]

Arithmetic intensity is:

\[
I = \frac{\text{FLOPs}}{\text{bytes moved}}.
\]

- High \(I\): computation tends to be compute-bound.
- Low \(I\): moving data tends to be the bottleneck.

This explains a foundational LLM-serving fact:

> **Prefill** is often more compute-heavy because it processes many prompt tokens together. **Decode** is often memory-bandwidth-bound because each next token requires repeatedly reading model weights and KV-cache data while doing comparatively little arithmetic.

The exact boundary depends on model, batch size, sequence length, precision, and hardware. But the prefill/decode asymmetry is the map behind much modern serving research.

---

# Part II — From text to GPT-style language models

## 2.1 “GPT” is an architectural family, not a full specification of current closed models

The original GPT line established the public template: a **decoder-only Transformer** trained autoregressively to predict the next token. GPT-3 demonstrated that scaling this setup produced strong few-shot behavior without task-specific parameter updates. [GPT-3 paper](https://arxiv.org/abs/2005.14165)

Current frontier GPT-family architectures are not fully public. Therefore, learn the canonical public decoder stack rather than pretending undocumented proprietary details are known.

```text
text
  → tokenizer
  → token IDs
  → token embeddings + position representation
  → repeated decoder blocks
  → vocabulary logits
  → softmax / decoding policy
  → generated tokens
```

## 2.2 Tokenization

Neural networks consume numbers, not character strings. A tokenizer maps text to a finite vocabulary of integer IDs.

```text
"unbelievable" → ["un", "believ", "able"] → [IDs]
```

### Why subword tokenization?

A word-level vocabulary cannot handle every spelling, name, code token, and language gracefully. Character-level models create very long sequences. Subword tokenizers occupy the pragmatic middle.

Common families:

- **BPE / byte-pair encoding:** repeatedly merges frequent symbol pairs.
- **WordPiece:** related subword method, historically used in BERT-style models.
- **Unigram / SentencePiece:** learns a probabilistic subword vocabulary.
- **Byte-level tokenization:** preserves arbitrary text by operating over bytes.

Tokenizer design changes practical behavior: code, multilingual text, rare words, prices, dates, and formatting can use very different token counts. Since context windows and API costs are token-based, tokenization is an engineering concern, not merely preprocessing.

**Learn:**
- [SentencePiece](https://github.com/google/sentencepiece)
- OpenAI `tiktoken` source and examples. https://github.com/openai/tiktoken
- Hugging Face Tokenizers. https://github.com/huggingface/tokenizers

## 2.3 Embeddings: turning IDs into vectors

Let vocabulary size be \(V\) and model width be \(d\). The embedding table is:

\[
E\in\mathbb{R}^{V\times d}.
\]

For token ID \(i\), lookup returns row \(E_i\). It is mathematically equivalent to multiplying a one-hot vector by \(E\), but implemented as an efficient table lookup.

The key nuance:

> An embedding is not a dictionary definition. It is a learned coordinate that becomes useful because the entire training objective rewards it for supporting accurate next-token prediction.

## 2.4 Positional information

Self-attention alone is permutation-equivariant: without positional information, it cannot distinguish `dog bites man` from `man bites dog` merely from token identities.

Common position approaches:

- learned absolute position embeddings;
- sinusoidal position embeddings;
- relative position schemes;
- **RoPE** (Rotary Position Embedding), which rotates query/key coordinate pairs according to position.

RoPE became common because relative-position effects arise naturally through dot products of rotated queries and keys. [RoFormer / RoPE paper](https://arxiv.org/abs/2104.09864)

### A small RoPE intuition

For each pair of hidden dimensions, position \(p\) applies a rotation:

\[
R(p)=
\begin{bmatrix}
\cos\theta_p & -\sin\theta_p\\
\sin\theta_p & \cos\theta_p
\end{bmatrix}.
\]

The model does not add a “position number” like a sticky note. It represents each query and key in a position-rotated geometry, so their relationship can depend on relative displacement.

## 2.5 The decoder-only Transformer block

The Transformer paper introduced attention-only sequence modeling without recurrence or convolutions. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

A modern GPT-style decoder block commonly has:

```text
hidden state x
  ├─ RMSNorm / LayerNorm
  ├─ causal multi-head self-attention
  ├─ residual addition
  ├─ RMSNorm / LayerNorm
  ├─ feed-forward network (often SwiGLU)
  └─ residual addition
```

Repeated blocks gradually transform a sequence of token vectors into contextual representations.

## 2.6 Self-attention, carefully

Given an input matrix \(X\in\mathbb{R}^{T\times d}\), the model constructs three projections:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
\]

For one attention head:

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.
\]

where \(M\) is the causal mask.

### What each term means

- \(Q\): what each token is asking for.
- \(K\): what each token offers as an address or label.
- \(V\): the information to mix when a key is selected.
- \(QK^T\): pairwise compatibility scores between token positions.
- \(\sqrt{d_k}\): stabilizes score scale.
- softmax: turns each row of scores into weights that sum to 1.
- \(M\): prevents a token from attending to future positions during causal language modeling.

### Causal masking

For token position \(t\), attention may inspect positions \(\leq t\), but not \(>t\). The future positions receive \(-\infty\) before softmax, resulting in probability zero.

```text
Position 4 may attend to: 1, 2, 3, 4
Position 4 may not attend to: 5, 6, 7, ...
```

This makes next-token training and left-to-right generation compatible.

## 2.7 Multi-head attention

Instead of one giant attention operation, the model uses multiple heads, each with its own query/key/value projections. Heads may learn different kinds of relations: syntax, nearby structure, long-range reference, code delimiters, or patterns that defy easy labels.

\[
\mathrm{MHA}(X)=\mathrm{Concat}(head_1,\dots,head_h)W_O.
\]

The key point is not that every head is human-interpretable. The point is that multiple low-dimensional attention subspaces are more expressive than a single one.

## 2.8 MLPs, SwiGLU, normalization, and residual streams

Attention routes information between positions. The feed-forward sublayer transforms information **within** each position.

A common gated MLP is **SwiGLU**:

\[
\mathrm{SwiGLU}(x)=(xW_1)\odot\mathrm{swish}(xW_2)W_3.
\]

The gate lets the network modulate which transformed features pass forward.

### RMSNorm

RMSNorm normalizes using root mean square magnitude without subtracting the mean:

\[
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\odot g.
\]

It is a common modern alternative to LayerNorm. [RMSNorm paper](https://arxiv.org/abs/1910.07467)

### Residual connections

A residual block looks like:

\[
x_{l+1}=x_l+f_l(x_l).
\]

This gives each layer a “write to the residual stream” role rather than forcing it to rebuild a representation from scratch. It also makes deep optimization substantially more stable.

## 2.9 From hidden state to next token

After the last decoder block, the hidden state at the final active position is projected to vocabulary logits:

\[
z_t=h_tW_U+b.
\]

Softmax turns logits into a probability distribution over tokens. A decoding strategy selects a token, appends it to the sequence, and repeats.

### Decoding choices

| Method | Idea | Trade-off |
|---|---|---|
| Greedy | choose highest-probability token | deterministic but can be dull or locally brittle |
| Temperature | rescale logits before sampling | higher temperature increases diversity and risk |
| Top-k | sample only from k most likely tokens | prevents very low-probability tails |
| Top-p / nucleus | sample smallest set whose mass exceeds p | adapts candidate count to uncertainty |
| Beam search | track multiple candidate sequences | often useful in constrained generation, less universal for open-ended chat |
| Constrained decoding | restrict output to schema/grammar | useful for JSON, tool calls, code formats |

## 2.10 Multi-query attention and grouped-query attention

The KV cache becomes expensive when every query head has separate keys and values.

- **MHA:** each head has distinct Q, K, V.
- **MQA:** many query heads share one K/V head.
- **GQA:** query heads share K/V groups, a middle ground.

GQA reduces KV-cache storage and memory traffic while preserving more expressiveness than pure MQA. [GQA paper](https://arxiv.org/abs/2305.13245)

This is a perfect example of architecture and systems co-design: the change is motivated not just by model quality but by **autoregressive inference economics**.

## 2.11 A small GPT pseudocode model

```python
class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def __call__(self, x, kv_cache=None):
        attn_out, new_kv = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv

class GPT:
    def __call__(self, token_ids, kv_cache=None):
        x = token_embedding(token_ids)
        x = apply_rope_through_attention(x)
        new_cache = []
        for block, layer_cache in zip(blocks, kv_cache_or_empty(kv_cache)):
            x, layer_cache = block(x, layer_cache)
            new_cache.append(layer_cache)
        logits = final_norm(x) @ unembedding.T
        return logits, new_cache
```

The exact production implementation is more complicated, but the conceptual structure is this compact.

---

# Part III — Architecture families beyond the dense Transformer

## 3.1 Why alternatives exist

Dense Transformers are powerful, but they create recurring costs:

- full attention is quadratic in sequence length during prefill;
- autoregressive decoding reads large weights repeatedly;
- KV cache grows with sequence length and concurrent requests;
- a dense MLP activates every parameter for every token;
- distributed clusters spend considerable time moving activations, gradients, and expert tokens.

The main alternatives change one of these facts.

```text
Dense Transformer:      every token uses all relevant dense blocks
MoE:                    every token uses a small selected set of experts
SSM / Mamba:            sequence state is updated recurrently rather than retained as a full KV cache
Linear attention:       compress or factor attention state to avoid quadratic work
Hybrid models:          keep attention where it is valuable; use cheaper sequence modules elsewhere
```

## 3.2 Mixture of Experts (MoE)

### The basic idea

An MoE layer replaces a single dense feed-forward network with many expert networks and a router. For each token, the router selects only the top \(k\) experts.

\[
g(x)=\mathrm{softmax}(W_rx),
\]

\[
\mathcal{E}(x)=\mathrm{TopK}(g(x),k),
\]

\[
y=\sum_{e\in\mathcal{E}(x)}g_e(x)f_e(x).
\]

The model can have enormous **total** parameter capacity while activating only a smaller subset per token.

### Total parameters versus activated parameters

This distinction is essential.

```text
Total parameters: all stored experts + shared layers
Activated parameters: the subset used for one token's forward pass
```

A model may have hundreds of billions of total parameters but only tens of billions activated per token. This can improve compute efficiency, but it does **not** make storage, routing, memory capacity, or distributed communication vanish.

### The real engineering problem: routing

The attractive story is “specialists.” The hard story is “load balancing.” If most tokens choose the same few experts:

- those experts become overloaded;
- other experts undertrain;
- devices hosting popular experts become stragglers;
- all-to-all communication becomes uneven;
- throughput collapses toward the slowest expert shard.

Common MoE concepts:

- **Top-k routing:** select a small fixed number of experts.
- **Capacity factor:** reserve per-expert token capacity to prevent overload.
- **Token dropping:** discard or reroute tokens beyond an expert’s capacity; cheap but potentially harmful.
- **Auxiliary load-balancing loss:** nudges router behavior toward balanced use.
- **Expert parallelism:** place experts across devices, requiring all-to-all token exchange.
- **Shared experts:** always-active experts that capture general patterns while routed experts specialize.

### Important papers

- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

DeepSeek-V3 is a useful modern systems case study: it reports 671B total parameters but 37B activated per token, alongside Multi-head Latent Attention, an auxiliary-loss-free load-balancing approach, multi-token prediction, and FP8-oriented engineering. Treat it as a technical report and architecture case study rather than a universal recipe. [DeepSeek-V3](https://arxiv.org/abs/2412.19437)

## 3.3 Multi-head latent attention (MLA)

MLA is an architectural approach to reduce KV-cache pressure by compressing key/value representations into a latent representation and reconstructing attention-related components as needed. It is important because it acknowledges that for long-context autoregressive inference, **the cache can matter as much as the model weights**.

Learn it after you understand ordinary MHA, MQA/GQA, and KV caching. The DeepSeek-V2/V3 technical materials are the primary case study.

## 3.4 State-space models and Mamba

### The continuous starting point

A classical linear state-space model can be written:

\[
h'(t)=Ah(t)+Bx(t),
\]

\[
y(t)=Ch(t)+Dx(t).
\]

After discretization for token steps:

\[
h_t=\bar{A}h_{t-1}+\bar{B}x_t,
\]

\[
y_t=Ch_t+Dx_t.
\]

The state \(h_t\) is a compressed memory of the past. Unlike a Transformer KV cache, its memory does not grow one stored token at a time in the same way.

### What Mamba changes

Earlier SSMs could struggle with content-dependent selection. Mamba makes key parameters input-dependent, allowing the model to selectively retain, update, or forget information depending on the current token. [Mamba paper](https://arxiv.org/abs/2312.00752)

The core conceptual contrast is:

```text
Attention:
  current token can directly compare against stored representations of many past tokens

Selective SSM:
  current token updates a recurrent state that carries selected information from the past
```

### Why Mamba is attractive

- linear scaling in sequence length for its sequence operation;
- no conventional growing attention KV cache;
- hardware-aware scan algorithms;
- potentially high throughput on very long sequences.

### Why it is not a free replacement

Attention provides direct content-addressable access to earlier token representations. A finite recurrent state can become an information bottleneck. The research question is not “Is Mamba better than Transformers?” It is:

> Under what sequence length, task distribution, hardware, training budget, and hybrid design does each mechanism provide the best quality-cost trade-off?

### Mamba-2 / state-space duality

Mamba-2 relates certain attention-like and state-space computations through **structured state-space duality** and introduces a faster core layer. [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)

### Mamba-3

The Mamba project now lists **Mamba-3** as a 2026 sequence-modeling research direction. It should be followed as active research, not treated as settled production architecture. [Mamba repository](https://github.com/state-spaces/mamba)

## 3.5 Hybrid Transformer–Mamba models

A hybrid model asks a pragmatic question: why force one primitive to do every job?

**Jamba** interleaves Transformer and Mamba layers and adds MoE in selected layers. It is valuable as a design pattern: attention for direct token interaction, state-space modules for efficient sequence processing, and sparse experts for capacity. [Jamba](https://arxiv.org/abs/2403.19887)

## 3.6 Linear attention and the renewed long-context race

Linear-attention approaches seek to avoid the \(O(T^2)\) attention matrix by rearranging or approximating computations, often using a compressed running state. The historical challenge was matching full attention’s quality and expressiveness.

**Kimi Linear** is an important 2025 technical report because it claims a hybrid architecture combining Kimi Delta Attention with Multi-head Latent Attention, reporting improved efficiency for long contexts and reduced KV-cache use under its evaluation setup. Read it as a current research case study, with the usual caution that one technical report does not settle an architecture class. [Kimi Linear](https://arxiv.org/abs/2510.26692)

## 3.7 Convolutions, recurrence, memory, and the architecture question

Architectures are converging rather than cleanly replacing one another.

- Attention offers flexible content addressing.
- MoE offers conditional capacity.
- SSMs offer compact recurrent state and favorable long-sequence scaling.
- Linear attention seeks efficient stateful alternatives.
- Convolutions offer local inductive bias and hardware-friendly locality.
- External memory and retrieval shift some information outside model activations.

A mature AI engineer does not learn these as rival sports teams. Learn their *state representation* and *data movement pattern*.

## 3.8 Long-context Transformer techniques

Not every long-context model abandons full attention. Common approaches include:

- **sliding-window attention:** each token attends locally, with selected global tokens or layers;
- **sparse attention patterns:** predefine which token pairs may interact;
- **context parallelism / ring attention:** distribute long sequences across devices;
- **position interpolation and RoPE scaling:** adapt position behavior beyond original training length;
- **retrieval and external memory:** avoid placing all knowledge directly in the prompt;
- **KV-cache compression/eviction:** retain only selected state at serving time.

The important distinction is between *supporting a large context window* and *reliably using information throughout that window*. Long-context evaluation must test retrieval, multi-hop reasoning, distraction robustness, and real application workflows, not only whether a system accepts a large token count.

A useful survey: [Advancing Transformer Architecture in Long-Context LLMs](https://arxiv.org/abs/2311.12351).

## 3.9 Diffusion and masked language-model generation

Autoregressive models generate left to right. Diffusion- and masked-generation language models explore different generation schedules, often iteratively denoising or filling multiple positions. Their promise is greater parallelism in generation; their challenge is matching autoregressive quality, controllability, and efficient variable-length decoding.

This is worth knowing as a frontier family, especially when considering generation latency and edit-style tasks. It is not yet a reason to discard the autoregressive mental model, which remains the most important serving pattern for present LLM systems.

---

# Part IV — Training foundation models

## 4.1 The training stack

```text
raw corpora
  → licensing / governance / filters
  → document normalization
  → deduplication
  → tokenizer training or selection
  → tokenized shards
  → distributed pretraining
  → checkpoints + evaluation
  → supervised fine-tuning
  → preference / RL / verifier-based post-training
  → deployment evaluation
```

Architecture is only one portion of model quality. Data quality, coverage, filtering, training stability, and post-training can dominate practical behavior.

## 4.2 Pretraining objective

A causal language model is trained on next-token prediction. It observes a prefix and predicts the next token.

```text
Input:  [The, cat, sat, on]
Target: [cat, sat, on, the]
```

Training is massively parallel because all token positions in a sequence can be scored simultaneously under the causal mask. Generation is sequential because the next unknown token depends on the previously generated one.

That asymmetry explains why **training** and **inference** are not merely the same workload at different scale.

## 4.3 Data engineering

Data pipeline decisions include:

- source licensing and provenance;
- language/domain mixture;
- quality filtering;
- deduplication at document, paragraph, and near-duplicate levels;
- contamination management for benchmark integrity;
- personal-data and sensitive-content handling;
- code/text/math mixture;
- synthetic-data generation and filtering;
- curriculum ordering;
- shard design for distributed I/O.

The model does not learn “the internet.” It learns the empirical distribution produced by the data pipeline. This is why data engineering is model engineering.

## 4.4 Synthetic data, distillation, and model self-improvement

Training data is no longer exclusively human-authored. Modern pipelines may use stronger teacher models to generate explanations, code, preference pairs, tool trajectories, or filtered synthetic tasks.

### Distillation

A teacher provides targets that are richer than a one-hot next-token label. A student can learn from:

- teacher logits or probability distributions;
- generated rationales and solutions;
- filtered tool-use trajectories;
- preference labels;
- self-play or verifier-selected samples.

Distillation transfers behavior, not guaranteed truth. A student can inherit a teacher’s blind spots, verbosity habits, and benchmark artifacts. The data filter and evaluator remain central.

### Why this matters operationally

Distillation often shifts cost from expensive inference-time reasoning to cheaper training-time imitation. But a model distilled from long reasoning traces may still need explicit training for concise answering, robust tool use, or calibrated uncertainty.

## 4.5 Scaling laws and compute-optimal training

Scaling laws model how loss changes with model parameters, data, and compute. The Chinchilla result argued that many earlier large language models were undertrained relative to their parameter count and that, under its studied regime, model size and training-token count should scale together. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

The enduring lesson is not “use exactly one token-to-parameter ratio forever.” It is:

> A training budget must be allocated jointly across model size, data volume, architecture, sequence length, and expected inference demand.

If a model will receive massive inference traffic, a smaller model trained longer may be economically attractive because inference costs recur for every request. [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)

## 4.6 Memory accounting during training

Training stores more than the model weights.

For a parameter count \(P\), a rough unsharded mixed-precision Adam-style accounting can include:

```text
BF16 model weights                  ≈ 2P bytes
BF16 gradients                      ≈ 2P bytes
FP32 master weights, if maintained  ≈ 4P bytes
Adam first moment                   ≈ 4P bytes
Adam second moment                  ≈ 4P bytes
-------------------------------------------
parameters + optimizer states       ≈ 16P bytes
+ activations, temporary buffers, communication buffers, fragmentation
```

This is a teaching estimate, not a sizing guarantee. Framework, optimizer, precision, checkpointing, sharding, and fused implementations alter it. But the key point is permanent:

> Optimizer states and activations often make training memory much larger than model-weight storage alone.

## 4.7 Distributed training parallelisms

### Data parallelism (DP)

Each device holds a model replica and processes different data. Gradients are synchronized, usually with all-reduce.

```text
GPU 1: batch A → gradients ┐
GPU 2: batch B → gradients ├─ all-reduce → same update
GPU 3: batch C → gradients ┘
```

### Fully sharded data parallelism / ZeRO

Instead of replicating all optimizer states, gradients, and parameters across every device, shard them across devices and gather pieces only when needed.

- **ZeRO-1:** shard optimizer state.
- **ZeRO-2:** shard optimizer state + gradients.
- **ZeRO-3:** shard optimizer state + gradients + parameters.

Resources:
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO guide](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

### Tensor parallelism (TP)

Split individual large matrix multiplications across devices. For example, shard a weight matrix by columns or rows, then use collective communication to combine results.

### Pipeline parallelism (PP)

Put different layers on different devices or device groups. Microbatches keep pipeline stages busy, though pipeline bubbles and communication become important.

### Context / sequence parallelism

Shard long sequences or some sequence-oriented operations to reduce activation-memory pressure and improve scaling for long context.

### Expert parallelism (EP)

Place MoE experts across devices. Tokens are routed to the devices holding their selected experts, often via all-to-all communication.

### The practical composition

Large systems often combine multiple forms:

```text
Data parallel × tensor parallel × pipeline parallel × context parallel × expert parallel
```

This is called **multi-dimensional parallelism**. [Megatron-LM scaling paper](https://arxiv.org/abs/2104.04473)

## 4.8 Communication is part of the model

Collectives matter:

- **all-reduce:** aggregate gradients and return result to all participants;
- **all-gather:** collect shards so each participant obtains the full logical tensor;
- **reduce-scatter:** reduce then distribute shards;
- **all-to-all:** every device exchanges different slices with every other device, central to MoE routing.

A model may be mathematically elegant but operationally poor if communication dominates compute.

## 4.9 Mixed precision training

Mixed precision uses lower-precision formats for appropriate operations while preserving enough high-precision state for stability.

Typical concepts:

- BF16 / FP16 activations and matrix multiplies;
- FP32 accumulations where necessary;
- loss scaling for FP16 regimes;
- FP8 recipes with scaling metadata and carefully designed kernels;
- numerical monitoring for overflow, NaNs, and loss spikes.

Modern accelerators increasingly support FP8 and FP4 pathways, but the recipe is hardware- and model-dependent. Do not reduce “mixed precision” to changing a dtype flag.

## 4.10 Checkpointing and fault tolerance

At scale, training is an operations problem:

- save model, optimizer, scheduler, tokenizer, and data-progress state;
- support resuming after preemption or machine failure;
- shard checkpoints to avoid one process becoming an I/O bottleneck;
- validate that resumed runs reproduce expected behavior;
- manage versioning of data and code.

A model checkpoint without its tokenizer, training configuration, data provenance, and evaluation context is an incomplete artifact.

---

# Part IV-A — Multimodality, retrieval, distillation, and evaluation

## 4A.1 Multimodal foundation models

A multimodal model connects representations from text, images, audio, video, documents, or sensor data. Common designs include:

```text
image / audio / video encoder
  → modality features
  → projector, resampler, or cross-attention bridge
  → language-model hidden space
  → autoregressive text or action output
```

Important architectural choices:

- **frozen encoder + learned projector:** economical adaptation path;
- **cross-attention:** language tokens attend to modality features;
- **unified token stream:** represent multiple modalities in a shared sequence;
- **late fusion:** separate encoders combine near the output;
- **native multimodal pretraining:** co-train modalities deeply from the beginning.

Engineering concerns include image tiling, video-frame sampling, audio token rates, multimodal KV-cache costs, OCR accuracy, grounding evaluation, and training-data rights. Multimodality is not just “put an image in the prompt.” It is a change to input representation, sequence length, serving memory, and evaluation.

## 4A.2 Retrieval and external memory

Retrieval-augmented generation moves some knowledge outside parameters and outside the current context until needed:

```text
query → retrieve evidence → rerank / filter → build grounded context → model response
```

Retrieval is useful when knowledge changes, sources must be cited, documents are too large for the context budget, or an application needs access control. It does not eliminate model error: poor retrieval, bad chunking, stale indexes, and unsupported synthesis remain failure modes.

For the companion agentic-engineering curriculum, retrieval becomes part of the harness and context middleware. In this guide, notice the systems impact: retrieval changes prompt length, prefix-cache behavior, prefill cost, and latency.

## 4A.3 Evaluation is an engineering subsystem

A benchmark score is not a deployment specification. Every model or runtime change needs an evaluation stack with:

- **capability evaluations:** task accuracy, code tests, tool success, reasoning;
- **reliability evaluations:** schema validity, retries, deterministic regression cases;
- **safety evaluations:** policy, privacy, jailbreak, misuse-specific tests;
- **system evaluations:** TTFT, TPOT, cost, peak memory, failure rate, tail latency;
- **data evaluations:** leakage, duplication, freshness, provenance;
- **human evaluation:** when mechanical verifiers are insufficient.

Treat evaluations as versioned software: fixed inputs, fixed decode configuration, traceable model/runtime versions, and clear promotion criteria.

---

# Part V — The physical substrate: CPU, GPU, TPU, LPU, memory, and networks

## 5.1 The hardware mental model

```text
                    high-level application / server
                                 │
          CPU: scheduling, networking, tokenization, orchestration
                                 │
             accelerator: massive numerical parallelism
                                 │
      registers / SRAM / caches → HBM → host memory → storage
                                 │
                  interconnect: NVLink / ICI / fabric / Ethernet
```

The important engineering fact is that a model’s data constantly moves through this hierarchy. The speed of arithmetic alone is not enough.

## 5.2 CPU: the conductor, not the obsolete ancestor

CPUs excel at:

- general-purpose control flow;
- operating-system interaction;
- tokenization and request preprocessing;
- scheduling and networking;
- data loading and storage I/O;
- branch-heavy logic;
- small-batch and latency-sensitive orchestration;
- hosting or offloading workloads when memory capacity matters more than raw throughput.

A CPU is not “slow.” It is optimized for different work: low-latency control, large caches, sophisticated branching, and generality.

## 5.3 GPU: throughput-oriented parallel computation

GPUs contain many parallel execution units and are designed to execute many similar numerical operations simultaneously. For AI, their value comes from:

- dense matrix multiplication throughput;
- Tensor Cores / matrix engines;
- HBM bandwidth;
- mature programming and kernel ecosystem;
- multi-GPU interconnects and collective communication libraries.

The CUDA programming model exposes thread hierarchies, memory spaces, and synchronization. Even if you never hand-write CUDA, understanding this hierarchy explains why kernel tiling, batching, and memory coalescing matter. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### GPU memory hierarchy, simplified

```text
Registers          tiny, fastest, thread-private
Shared memory      fast SRAM, cooperative within a block
L1 / L2 cache      hardware-managed reuse layers
HBM                large, high-bandwidth accelerator memory
Host DRAM          larger but slower and farther away
NVMe / storage     massive but far slower
```

Good kernels maximize reuse of data after loading it from HBM into nearby memory. FlashAttention is a canonical example.

## 5.4 Tensor Cores and matrix engines

Matrix multiplication is the dominant operation in dense neural networks. Specialized matrix engines accelerate tiled multiply-accumulate operations such as:

\[
C \leftarrow AB+C.
\]

They typically support multiple numeric formats. The practical lesson:

> Peak FLOP numbers matter only when your shapes, precision, memory access, and kernel implementation keep the matrix engines fed.

## 5.5 TPU: purpose-built matrix computation at cluster scale

Google TPUs use Matrix Multiply Units (MXUs), vector units, HBM, and high-speed inter-chip interconnects. TPUs are designed around dense tensor computation and compilation-oriented execution. [Google TPU architecture documentation](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)

A TPU is not simply “a Google GPU.” Its programming and execution ecosystem has different assumptions, especially around XLA/JAX compilation, SPMD partitioning, and pod-scale topology.

### Why systolic arrays matter

A systolic array pipelines matrix multiplication so data flows through an array of multiply-accumulate units, allowing high reuse of operands. This is especially efficient for regular dense linear algebra.

## 5.6 Groq LPU: latency-oriented, compiler-scheduled inference hardware

Groq uses the term **Language Processing Unit (LPU)** for its inference-oriented architecture. Groq describes it as a compiler- and software-defined single-core design with on-chip SRAM and deterministic, token-based execution. [Groq LPU architecture](https://groq.com/lpu-architecture)

The educational point is not to accept any vendor’s headline throughput number as universally comparable. It is to understand the architectural bet:

```text
GPU-style philosophy:
  general parallel hardware + dynamic scheduling + broad workload flexibility

LPU-style philosophy:
  highly scheduled, model-specific dataflow + predictable execution + low-latency token generation
```

This can be attractive for known inference workloads where compilation, static scheduling, and data locality can be exploited. The trade-off is typically less generality than a broad GPU ecosystem.

## 5.7 Other accelerator families worth knowing

| Family | Core idea | Why it matters |
|---|---|---|
| NVIDIA GPUs | mature CUDA ecosystem, tensor compute, NVLink | dominant broad ecosystem for training and inference |
| AMD Instinct | CDNA accelerators, ROCm ecosystem, high HBM capacity | important alternative platform for large model training/inference |
| Google TPU | MXU + XLA + pod-scale interconnect | strong example of hardware/compiler co-design |
| AWS Trainium | training-focused NeuronCore hardware | cloud-specific training alternative |
| AWS Inferentia | inference-focused Neuron hardware | cloud-specific deployment path |
| Groq LPU | deterministic compiled token inference | latency-oriented architecture case study |
| Cerebras WSE | wafer-scale compute | illustrates minimizing inter-device communication through enormous on-chip scale |
| Intel Gaudi | accelerator with Ethernet-oriented scale-out | illustrates alternative networking and accelerator-stack design |
| FPGA / custom ASIC | hardware tailored to fixed or edge workloads | valuable when latency, power, or deployment constraints dominate |

Live vendor sources:

- [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/)
- [AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)
- [AWS Neuron stack](https://aws.amazon.com/ai/machine-learning/neuron/)
- [AMD Instinct MI300 architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [Google Cloud TPU documentation](https://cloud.google.com/tpu/docs/intro-to-tpu)

## 5.8 Memory capacity versus memory bandwidth

These are different constraints.

- **Capacity** asks: can the weights, activations, and KV cache fit?
- **Bandwidth** asks: can the processor read/write what it needs fast enough?

For LLM decode, a system can have enough memory capacity but still produce tokens slowly because it must repeatedly stream large weights from HBM.

For long-context serving, the KV cache can fit only with aggressive memory management, quantization, offloading, or architecture changes.

## 5.9 Interconnects and clusters

Scaling beyond one accelerator introduces data movement between chips and nodes.

Important concepts:

- **PCIe:** common host-to-device and some device-to-device pathway.
- **NVLink / NVSwitch:** NVIDIA interconnect and switching fabric for high-bandwidth GPU communication.
- **Infinity Fabric:** AMD interconnect family.
- **ICI:** Google TPU inter-chip interconnect.
- **InfiniBand / RoCE Ethernet:** network fabrics for multi-node training and serving.
- **RDMA:** remote direct memory access, reducing CPU involvement in network transfers.

When an MoE router sends tokens to experts on other GPUs, or tensor parallelism performs all-reduce, the network is literally on the critical path of the model’s forward pass.

## 5.10 Hardware evaluation: do not compare only peak TOPS

Compare hardware for a specific workload using:

- precision actually supported by the model and kernels;
- HBM capacity and bandwidth;
- single-request latency;
- batched throughput;
- interconnect topology;
- compiler/runtime maturity;
- supported model architectures;
- quantization support;
- power and rack constraints;
- procurement and operational cost;
- debugging and observability tooling.

Peak compute is a menu photo. Your workload is the meal.

## 5.11 Compilers, kernels, and the software layer closest to hardware

The model graph you write in Python is not the final program that runs on silicon. A compiler/runtime stack lowers operations into optimized kernels and schedules memory movement.

```text
PyTorch / JAX model
  → graph capture / optimization
  → fused kernels and GEMMs
  → accelerator-specific code
  → driver and hardware execution
```

Important pieces:

- **CUDA:** NVIDIA’s programming and runtime ecosystem.
- **ROCm/HIP:** AMD’s GPU software ecosystem.
- **XLA:** compiler stack used heavily by JAX and TPU workflows.
- **Triton:** Python-based language/compiler for writing custom DNN kernels. [Triton](https://triton-lang.org/)
- **TensorRT / TensorRT-LLM:** NVIDIA inference compilation/runtime stack. [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
- **TorchInductor:** PyTorch compiler path.
- **TVM / MLIR-style systems:** compiler research and deployment infrastructure.

### Kernel fusion

A sequence such as `bias → activation → dropout → residual add` may be fused into one kernel so intermediate results do not repeatedly leave fast local memory and return to HBM. Fusion reduces launch overhead and memory traffic.

### Why a custom kernel can matter

For bandwidth-bound operations, reducing extra reads/writes can dominate any arithmetic trick. The Triton fused-softmax tutorial is a good first example of this principle. [Fused Softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

---

# Part VI — Inference and serving systems

## 6.1 The two phases: prefill and decode

### Prefill

The system processes the input prompt, computes activations for all prompt tokens, and writes keys/values into the KV cache.

```text
Prompt: 2,000 tokens
→ run model on all 2,000 tokens
→ build cache for each layer
→ first output-token distribution
```

### Decode

The system generates one or more next tokens, using the cache instead of recomputing all past attention projections.

```text
cache + newest token
→ model forward pass for current step
→ next token
→ append its K/V to cache
→ repeat
```

### Why the phases behave differently

- Prefill has larger matrix multiplications and can be highly parallel.
- Decode has tiny token increments and repeatedly reads model weights and cache state.
- Therefore prefill often trends compute-heavy, while decode frequently trends memory-bandwidth-heavy.

This is the source of prefill/decode disaggregation research and phase-specialized hardware ideas.

## 6.2 KV cache: the hidden memory bill

For each transformer layer, the model stores keys and values for prior tokens. A rough cache-size estimate is:

\[
\text{KV bytes}\approx 2\times L\times T\times n_{kv}\times d_h\times b,
\]

where:

- \(L\): number of layers,
- \(T\): cached tokens,
- \(n_{kv}\): number of key/value heads,
- \(d_h\): head dimension,
- \(b\): bytes per stored value,
- factor 2: keys plus values.

Multiply by batch size / concurrent sequences. This is why GQA, MLA, KV-cache quantization, paging, offloading, eviction policy, prefix sharing, and long-context architecture choices matter.

## 6.3 The production metrics

| Metric | Meaning | What affects it |
|---|---|---|
| TTFT | time from request arrival to first token | queueing, prefill, prefix cache, scheduling |
| TPOT | average time between generated tokens | decode kernel, weights, KV cache, batch behavior |
| End-to-end latency | request arrival to final token | TTFT + generated length × TPOT + queueing |
| Throughput | aggregate tokens or requests per second | batching, hardware utilization, model size |
| Goodput | work meeting SLO targets | scheduler quality, tail latency, capacity |
| GPU utilization | how fully compute/memory are used | kernel efficiency and batch composition |
| Cost/token | infrastructure cost per generated token | hardware, utilization, precision, routing |

For agentic systems, also track:

- prefix-cache hit rate;
- tool-call latency;
- average context length and output length;
- cancellation rate;
- failed structured-output rate;
- reasoning-token budget;
- queue delay percentile.

## 6.4 Continuous batching

Traditional static batching waits for a fixed batch, processes it, then starts another. Autoregressive requests have different prompt and output lengths, so static batching wastes capacity.

**Continuous batching** dynamically admits new requests as existing sequences finish or as memory becomes available.

```text
Time 0: requests A, B, C begin
Time 1: A generates token; D arrives and can join
Time 2: B finishes; E joins
Time 3: C continues; D and E continue
```

The scheduler is now an important part of model performance.

## 6.5 PagedAttention and vLLM

The KV cache resembles dynamic memory allocation: requests have different lengths, may branch in beam search, may be canceled, and may share prefixes.

**PagedAttention** stores KV cache in fixed-size blocks rather than requiring each sequence to occupy one contiguous allocation. This reduces fragmentation and enables more flexible cache management. [vLLM / PagedAttention paper](https://arxiv.org/abs/2309.06180)

[vLLM](https://github.com/vllm-project/vllm) is a major open-source serving engine. Its current project materials list features including PagedAttention, continuous batching, chunked prefill, prefix caching, CUDA/HIP graphs, and many quantization formats. Treat individual implementation details as version-sensitive because serving stacks evolve quickly.

## 6.6 RadixAttention and SGLang

Prefix reuse becomes powerful when requests share system prompts, conversation prefixes, documents, retrieval context, or tool schemas.

SGLang’s RadixAttention organizes cache reuse around shared prefixes. Its runtime also exposes modern serving capabilities such as prefix caching, continuous batching, speculative decoding, prefill/decode disaggregation, multi-GPU parallelism, structured outputs, and quantization. [SGLang repository](https://github.com/sgl-project/sglang) and [documentation](https://docs.sglang.ai/)

The broader lesson:

> In agent systems, a good cache key may be as valuable as a faster matrix multiplication.

## 6.7 FlashAttention: optimize I/O, not just FLOPs

Naïve attention materializes or repeatedly moves large intermediate matrices. FlashAttention uses tiling and recomputation to reduce expensive HBM reads/writes while computing exact attention. [FlashAttention](https://arxiv.org/abs/2205.14135)

This is a landmark systems insight:

```text
Same mathematical attention
≠ same hardware cost
```

FlashAttention-2 improves work partitioning and parallelism to better utilize GPUs. [FlashAttention-2](https://arxiv.org/abs/2307.08691)

## 6.8 Prefix caching and prompt caching

If many requests begin with an identical prefix:

```text
same system prompt
+ same tool descriptions
+ same repository instructions
+ same large document header
```

then compute and KV cache for that prefix can be reused. This is especially valuable for agent systems, customer-support flows, and repeated document-analysis templates.

Beware: cache design creates correctness and privacy responsibilities. A prefix cache must never leak one tenant’s state into another tenant’s request.

## 6.9 Speculative decoding

Autoregressive decoding is sequential. Speculative decoding tries to break this bottleneck:

1. a fast drafter proposes several future tokens;
2. the large target model verifies them in a parallelized forward pass;
3. accepted tokens are emitted; mismatches trigger correction.

When implemented with correct acceptance logic, speculative decoding can preserve the target model’s output distribution while reducing sequential target-model steps.

Important references:

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE](https://arxiv.org/abs/2401.15077)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)

### Important caution

Speedup depends on acceptance rate, drafter cost, batch shape, model quantization, hardware, and decoding setting. Combining two optimizations does not guarantee their gains multiply. Research on speculative decoding plus quantization explicitly finds interaction effects that can reduce expected gains. [Speculative Decoding Meets Quantization](https://arxiv.org/abs/2505.22179)

## 6.10 Prefill/decode disaggregation

Because prefill and decode have different bottlenecks, a serving system can place them on distinct pools of hardware or scheduling paths.

```text
request
  → prefill workers: prompt processing / cache construction
  → transfer or share KV state
  → decode workers: token streaming
```

Potential benefits:

- isolate compute-heavy prompt bursts from memory-heavy decoding;
- tune hardware and batch policies per phase;
- improve tail latency and cluster utilization.

Costs:

- KV-cache transfer / ownership complexity;
- scheduling complexity;
- extra network or device-to-device traffic;
- difficult autoscaling and state placement.

Current research treats this as a central serving direction, not a solved checkbox.

## 6.11 Model parallel inference

Large models often require multiple accelerators.

- **Tensor parallelism:** split each layer’s math across accelerators; fast links are important.
- **Pipeline parallelism:** split layers across stages; can help capacity but adds pipeline behavior.
- **Expert parallelism:** send tokens to expert-hosting devices.
- **Data parallel replicas:** run separate model replicas for serving capacity.

The right split depends on model architecture, batch size, sequence length, interconnect, and latency target.

## 6.12 Serving stack anatomy

```text
API gateway / auth / quotas
  → request normalization and tokenization
  → router (model, replica, region, priority)
  → scheduler (queue, batch, cache-aware placement)
  → model runtime (kernels, KV cache, sampling)
  → accelerator workers
  → token streaming
  → telemetry, traces, billing, safety filters
```

This is why “deploy an LLM” is an infrastructure problem, not simply `model.generate()` behind an HTTP endpoint.

## 6.13 Runtime choices: a practical map

| Runtime / tool | Best thought of as | Notes |
|---|---|---|
| vLLM | open, general high-throughput serving engine | PagedAttention lineage; broad model/runtime support |
| SGLang | high-performance structured generation and serving runtime | strong prefix/radix-cache and agentic-workload relevance |
| TensorRT-LLM | NVIDIA-optimized inference compilation/runtime | in-flight batching, paged KV cache, quantization, multi-GPU support |
| Hugging Face TGI | deployment toolkit | currently in maintenance mode; still a useful reference implementation |
| llama.cpp / GGUF | local and edge-oriented C/C++ runtime ecosystem | model-format conversion and broad CPU/GPU backend story |
| Transformers | model-library and prototyping layer | increasingly includes continuous-batching capabilities, but production needs workload testing |

Resources:

- [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [Hugging Face TGI docs](https://huggingface.co/docs/text-generation-inference/index)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

The selection rule is not brand loyalty. Match the runtime to model architecture, hardware, latency target, quantization format, multi-tenancy requirement, and operational maturity.

## 6.14 Inference research you should watch

- cache-aware scheduling and placement;
- mathematical optimization for LLM-serving scheduling, routing, and eviction;
- prefill/decode disaggregation;
- speculative methods that do not require a fully independent draft model;
- low-bit kernel co-design;
- MoE routing and expert-cache locality;
- long-context cache compression and eviction;
- CPU offload and multi-tier memory;
- deterministic serving and reproducibility under batching;
- agent-specific serving where tools, branching, and repeated prefixes dominate workload shape.

---

# Part VII — Quantization, compression, and efficient adaptation

## 7.1 The problem quantization solves

A model’s weights, activations, and cache consume memory and bandwidth. Quantization represents them with fewer bits.

```text
FP16 weight: 16 bits
INT8 weight:  8 bits
INT4 weight:  4 bits
```

At the simplest level, symmetric quantization maps a real value \(x\) to an integer \(q\):

\[
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right),q_{min},q_{max}\right),
\]

\[
\hat{x}=s\cdot q.
\]

Here \(s\) is a scale. As bit-width falls, the storage cost decreases but rounding error rises.

## 7.2 Quantization axes

You must always ask **what** is quantized.

| Target | Why quantize it? | Main risk |
|---|---|---|
| Weights | reduce model storage and weight bandwidth | outlier channels, error accumulation |
| Activations | lower runtime memory and enable low-bit GEMMs | activation outliers and dynamic range |
| KV cache | reduce long-context / concurrency memory | long-context degradation, cache sensitivity |
| Gradients | reduce distributed training communication | optimization instability |
| Optimizer state | reduce training memory | optimizer-quality trade-off |

And **how granularly** it is quantized:

- per-tensor;
- per-channel;
- per-group;
- per-token;
- blockwise / microscaled.

More granular scales can preserve accuracy but add metadata and kernel complexity.

## 7.3 PTQ versus QAT

### Post-training quantization (PTQ)

Quantize a trained model after training, usually using a calibration dataset or weight statistics.

Pros: cheap, practical, no full retraining.

Cons: accuracy can fall at low bit widths; quality depends on model, task, and kernel support.

### Quantization-aware training (QAT)

Simulate quantization during training or fine-tuning so the model adapts.

Pros: can improve low-bit quality.

Cons: more expensive and operationally complex.

## 7.4 Important methods

### GPTQ

GPTQ is a one-shot, post-training **weight quantization** method using approximate second-order information to make 3- and 4-bit generative models more accurate. [GPTQ](https://arxiv.org/abs/2210.17323)

### AWQ

Activation-aware Weight Quantization identifies salient weight channels and protects them during quantization. It is widely used in practical low-bit inference stacks. [AWQ](https://arxiv.org/abs/2306.00978)

### SmoothQuant

SmoothQuant addresses activation outliers by mathematically migrating some quantization difficulty from activations to weights, enabling W8A8 pathways in appropriate hardware/software environments. [SmoothQuant](https://arxiv.org/abs/2211.10438)

### QLoRA and NF4

QLoRA is mainly an **efficient fine-tuning** technique rather than a generic inference method. It keeps a pretrained model frozen in 4-bit form, trains low-rank adapters, and introduced NF4 plus double quantization for memory efficiency. [QLoRA](https://arxiv.org/abs/2305.14314)

### KV-cache quantization

KV cache is increasingly important for long context and high concurrency. Methods may quantize it to 8, 4, or fewer bits, sometimes adaptively by layer, head, or token. This needs separate evaluation because a model with good weight-only quantization may still degrade under low-bit cache storage.

## 7.5 Formats: INT4, NF4, FP8, FP4, MXFP4, NVFP4

Do not treat format names as a ranking.

- **INT4:** four-bit integers plus scales/zero points.
- **NF4:** four-bit format designed around normal-distribution assumptions, commonly associated with QLoRA.
- **FP8:** 8-bit floating formats with larger dynamic range than integers of similar bit width.
- **FP4:** 4-bit floating formats; precision/range trade-off is severe and hardware-specific.
- **Microscaling formats:** blocks share scaling factors to manage dynamic range.
- **NVFP4:** NVIDIA’s FP4-oriented format and scaling approach for Blackwell-class systems.

Hardware support is decisive. A format that compresses weights but lacks fast kernels can be slower than a higher-bit alternative.

## 7.6 Quantization is a systems decision

The quality-cost outcome depends on the entire stack:

```text
quantized checkpoint
  + calibration method
  + model architecture
  + context length
  + serving runtime
  + hardware kernel
  + batch shape
  + sampling / speculative strategy
  = real deployment behavior
```

A 4-bit model is not automatically 4× faster. It may be memory-lighter but compute-limited; dequantization overhead may matter; unsupported kernels may fall back to slower paths; speculative decoding may shift the bottleneck.

## 7.7 Practical selection guide

| Goal | Start with | Verify |
|---|---|---|
| simplest server-side reduction | BF16/FP16 → FP8 or W8A8 if supported | task quality, runtime kernel path |
| run large model on smaller GPU | weight-only 4-bit GPTQ/AWQ | real tokens/sec, memory headroom, quality |
| long context / high concurrency | KV quantization + GQA/MLA + paging | long-context retrieval/quality and cache stability |
| fine-tune model on constrained hardware | QLoRA | adapter quality, training stability |
| edge / CPU deployment | GGUF-style quantized runtimes, architecture-specific kernels | time-to-first-token, power, RAM usage |
| frontier hardware deployment | vendor-supported FP8/FP4 pathway | calibration, model-specific accuracy, deployment reproducibility |

## 7.8 Compression beyond quantization: pruning and distillation

Quantization is not the only way to reduce deployment cost.

- **Pruning:** remove weights, heads, channels, experts, layers, or tokens. Unstructured sparsity can be hard for hardware to exploit; structured sparsity is usually easier to accelerate.
- **Distillation:** train a smaller student to imitate a larger teacher’s outputs or trajectories.
- **Early exit / adaptive depth:** allow easier inputs to use less compute, but adds calibration and serving complexity.
- **Speculative generation:** an inference-time form of conditional compute, already covered in Part VI.

All forms of compression should be evaluated as end-to-end systems. A model that is smaller on disk but unsupported by production kernels has not necessarily been compressed in the way the user experiences.

## 7.9 Evaluation protocol for quantized models

Do not ship after one generic benchmark. Evaluate:

1. core task quality;
2. structured-output validity;
3. multilingual and code tasks if relevant;
4. long-context behavior;
5. tool-use behavior if agentic;
6. safety behavior;
7. TTFT, TPOT, throughput, memory, and cost;
8. output distribution or regression tests under fixed decoding;
9. failure modes at high concurrency;
10. calibration and reproducibility across model versions.

Useful surveys and evaluation work:

- [A Survey of Low-bit Large Language Models](https://arxiv.org/abs/2409.16694)
- [Evaluating Quantized Large Language Models](https://arxiv.org/abs/2402.18158)
- [Benchmarking Post-Training Quantization of LLMs](https://arxiv.org/abs/2601.09555)

---

# Part VIII — Reinforcement learning and post-training

## 8.1 Why pretraining is not enough

Next-token prediction teaches broad statistical competence. It does not directly optimize for:

- following instructions;
- useful dialogue behavior;
- truthfulness under uncertainty;
- formatting and tool-use conventions;
- refusing unsafe requests;
- long-horizon reasoning;
- code that passes tests;
- preference trade-offs;
- efficient reasoning length.

Post-training shapes behavior after pretraining.

## 8.2 The post-training landscape

```text
base pretrained model
  → supervised fine-tuning (SFT)
  → preference optimization / reward modeling
  → RLHF, RLAIF, DPO, PPO, GRPO, RLVR, etc.
  → domain- and tool-specific evaluation
```

Different methods solve different supervision problems. Avoid treating them as a chronological replacement ladder.

## 8.3 Reinforcement learning basics

An RL problem has:

- state \(s\),
- action \(a\),
- policy \(\pi_\theta(a\mid s)\),
- reward \(r\),
- trajectory \(\tau\),
- expected return \(J(\theta)\).

For language models:

```text
state      = prompt + generated prefix + possibly tool/environment state
action     = next token, tool call, or structured action
policy     = language model distribution
trajectory = full completion / reasoning trace / agent rollout
reward     = preference score, verifier result, test pass, environment outcome
```

The policy-gradient family seeks to improve expected reward:

\[
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a\mid s)A(s,a)\right],
\]

where \(A\) is an advantage estimate: “how much better was this action than a baseline?”

## 8.4 Supervised fine-tuning (SFT)

SFT trains on demonstrations:

```text
instruction → desired response
```

It is simple, stable, and often foundational. Its main limitation is that it learns from the distribution of demonstrations rather than directly optimizing a reward over generated rollouts.

## 8.5 RLHF

A common RLHF pipeline:

1. collect preference comparisons between candidate responses;
2. train a reward model that predicts preference;
3. optimize the language-model policy against the reward while constraining divergence from a reference policy.

The original InstructGPT work is canonical. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### Why the reference-model constraint exists

Without a constraint, the policy can exploit reward-model weaknesses or drift far from the pretrained language distribution. A KL penalty acts as a tether.

## 8.6 PPO

Proximal Policy Optimization constrains policy updates by clipping probability-ratio changes. A simplified objective is:

\[
L^{CLIP}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\;\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right],
\]

where

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
\]

PPO is powerful but operationally demanding for LLMs because it involves rollouts, reward estimation, advantage computation, policy updates, and careful distributed systems work.

- [PPO paper](https://arxiv.org/abs/1707.06347)

## 8.7 DPO: direct preference optimization

DPO converts preference pairs into a classification-style objective without training an explicit reward model and without online RL rollouts during fine-tuning. [DPO](https://arxiv.org/abs/2305.18290)

It is often appealing when you have preference data and want a simpler, stable optimization path. It does not replace RL when online environment interaction, tool feedback, exploration, or verifiable outcomes are central.

## 8.8 RL with verifiable rewards (RLVR)

RLVR uses rewards that can be mechanically checked:

- math answer matches an evaluator;
- code passes tests;
- theorem proof verifies;
- tool action produces a correct environment state;
- structured output satisfies a strict validator.

This can be more scalable and less subjective than human preference labels in domains with reliable verifiers.

### Why it matters

A good verifier changes the training problem. Rather than asking humans to rank every reasoning trace, you can generate multiple rollouts and reward outcomes automatically.

### The warning

A reward being “verifiable” does not mean it is fully aligned. It may be incomplete, exploitable, sparse, or accidentally correlated with desired behavior.

Research on spurious rewards in RLVR reports that apparent reasoning gains can sometimes arise even with rewards weakly or negatively correlated with the intended answer, which is an important warning against simplistic causal stories about how RLVR works. [Spurious Rewards](https://arxiv.org/abs/2506.10947)

## 8.9 GRPO

**Group Relative Policy Optimization** samples multiple completions for a prompt and estimates relative advantage from the group’s rewards, reducing dependence on a separate learned value model. It became prominent through DeepSeekMath and related reasoning-RL work. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

A simplified group-normalized advantage is:

\[
A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})+\epsilon}.
\]

The exact objective and implementation details matter, but this equation gives the intuition: a completion is judged relative to other attempts for the same prompt.

## 8.10 Reasoning RL and inference-time compute

Recent reasoning models combine two ideas:

1. improve the policy through post-training, including RL;
2. spend more inference-time compute through longer rollouts, self-checking, tool use, search, or multiple candidates.

This reframes model capability as not only “what is stored in weights?” but also “what computation can be spent at test time under a verifier or selection policy?”

The production issue is obvious: long reasoning traces and repeated rollouts are expensive. The inference system and the learning system become tightly coupled.

## 8.11 Agent reinforcement learning

In agent RL, the environment includes tools, files, browsers, code repositories, APIs, simulators, or games.

```text
policy proposes action
  → environment executes action
  → observation changes
  → agent continues
  → terminal verifier/reward
```

The reward is often sparse because a whole task either succeeds or fails. Useful research directions include:

- guided rollouts;
- curriculum generation;
- process rewards;
- test-time search;
- environment feedback shaping;
- self-play;
- trajectory reuse;
- offline-to-online adaptation.

[Agent-RLVR](https://arxiv.org/abs/2506.11425) is a useful software-engineering case study: it combines unit-test outcomes with guidance and environment feedback to make RLVR more usable for longer-horizon agent tasks.

## 8.12 DAPO and reproducible large-scale reasoning RL

DAPO is an open-source 2025 system and algorithmic recipe for large-scale LLM RL. It emphasizes issues that are highly practical in long chain-of-thought optimization: clipping behavior, dynamic sampling, token-level policy-gradient loss, and overlong reward shaping. [DAPO](https://arxiv.org/abs/2503.14476)

The important learning lesson is not to memorize DAPO’s name. It is to see that scalable LLM RL is **a full system**:

```text
prompt sampling
+ distributed rollouts
+ verifier / reward pipeline
+ policy updates
+ reference model / KL logic
+ checkpointing
+ evaluation
+ resource scheduling
```

## 8.13 RL failure modes

- **Reward hacking:** maximize the proxy while missing the true goal.
- **Sparse rewards:** almost all trajectories fail, yielding weak learning signal.
- **Entropy collapse:** outputs become overly narrow or repetitive.
- **Length hacking:** model emits unnecessary reasoning to exploit reward or avoid penalties.
- **Verifier bugs:** training hardens exploitation of an imperfect checker.
- **Distribution shift:** reward works in training but not deployment.
- **Mode collapse / loss of generality:** narrow task gains damage broad helpfulness.
- **Evaluation leakage:** test patterns leak into training or synthetic data.

The cure is not “more RL.” It is robust environment design, reward auditing, held-out evaluation, and rollback discipline.

---

# Part IX — Current research frontier, 2025–June 2026

> **How to read this section:** These are active directions, not settled doctrine. A technical report may demonstrate an important result under a particular model, benchmark, and hardware setup without becoming a universal production rule. Always inspect the evaluation setting, baseline, cost accounting, and released code.

## 9.1 Architecture frontier

### Sparse capacity plus inference-aware attention

- **DeepSeek-V3** combines MoE, Multi-head Latent Attention, load balancing without an auxiliary loss, multi-token prediction, and FP8-oriented training engineering. It is a strong example of architecture decisions designed around training and inference economics. [Paper](https://arxiv.org/abs/2412.19437)

### State-space / attention unification

- **Mamba-2** develops the structured state-space duality framework, drawing a closer theoretical and algorithmic connection between certain attention and state-space computations. [Paper](https://arxiv.org/abs/2405.21060)
- **Mamba-3** is listed as an active 2026 follow-on in the official Mamba repository. Follow the paper and code as the research matures. [Repository](https://github.com/state-spaces/mamba)

### Stronger linear-attention claims

- **Kimi Linear** claims a hybrid linear-attention architecture that outperforms full attention under its reported comparisons while reducing KV-cache use and increasing long-context decode throughput. This is a major result to scrutinize closely because linear attention has historically struggled to match full attention broadly. [Paper](https://arxiv.org/abs/2510.26692)

### Dynamic local operators

- 2026 work on dynamic short convolutions explores reintroducing input-dependent local operators as a complement to attention. This is a reminder that future architectures may be hybrid mosaics, not a single victorious primitive. [Paper](https://arxiv.org/abs/2606.03825)

## 9.2 Inference frontier

### Serving as mathematical optimization

A 2026 position paper argues that LLM serving increasingly needs explicit optimization methods for routing, scheduling, cache management, and SLO control rather than generic heuristics alone. [LLM Serving Needs Mathematical Optimization](https://arxiv.org/abs/2605.01280)

### Prefill/decode specialization

Research increasingly treats prefill/decode disaggregation as a systems primitive, since the phases use compute, memory, and scheduling differently. Specialized prefill/decode hardware proposals push the idea further: provision compute-heavy hardware for prefill and bandwidth-oriented hardware for decode. [SPAD](https://arxiv.org/abs/2510.08544)

### Speculative decoding is evolving

- **EAGLE-3** moves from feature prediction to direct token prediction with multi-layer feature fusion. [Paper](https://arxiv.org/abs/2503.01840)
- **P-EAGLE**, described by AWS in 2026, explores parallel speculative drafting to remove some sequential drafting overhead. Treat vendor benchmarks as workload-specific, but follow the method direction. [AWS technical post](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)

### Cache and state are becoming first-class infrastructure

Expect continued work on:

- KV-cache locality and placement;
- cache sharing across repeated agent prefixes;
- storage-bandwidth bottlenecks in long agent trajectories;
- CPU/off-GPU cache tiers;
- cache quantization;
- prefix-aware scheduling;
- deterministic execution.

## 9.3 Quantization frontier

### FP4 and micro-scaling

New accelerator generations expose FP4 and microscaling formats. The research question is shifting from “Can a model be quantized?” to “Which operands, groups, layers, and cache states can use which format under which kernel?”

- [RaZeR: NVFP4 quantization](https://arxiv.org/abs/2501.04052)
- [Comprehensive low-bit LLM survey](https://arxiv.org/abs/2409.16694)

### KV-cache quantization needs its own evaluation

Recent evaluation work warns that even 8-bit KV-cache quantization is not risk-free and robustness varies by method. [Benchmarking PTQ of LLMs](https://arxiv.org/abs/2601.09555)

### Quantization and speculative decoding interact

Quantization can reduce the bandwidth advantage that speculative decoding tries to exploit, because speculative verification may introduce extra compute. The correct system choice depends on the new bottleneck after quantization. [Speculative Decoding Meets Quantization](https://arxiv.org/abs/2505.22179)

## 9.4 Reinforcement-learning frontier

### RLVR is powerful, but theory and practice are still unsettled

- **RLVR analysis** studies how verifiable rewards can incentivize correct reasoning trajectories and argues for metrics beyond naïve pass@k interpretations. [Paper](https://arxiv.org/abs/2506.14245)
- **Spurious Rewards** warns that strong benchmark gains can appear with rewards that are not meaningfully aligned to the target answer, complicating simple explanations of reasoning-RL success. [Paper](https://arxiv.org/abs/2506.10947)

### Better exploration and curriculum

- **Scaf-GRPO** studies the “learning cliff,” where difficult prompts yield all-zero rewards and hence little gradient, and uses staged guidance to help models reach learnable trajectories. [Paper](https://arxiv.org/abs/2510.19807)
- **Rare-event prompt efficiency** studies how to select prompts that produce informative RL gradients rather than merely high reward variance. [Paper](https://arxiv.org/abs/2602.03452)

### RL for real agents

- **Agent-RLVR** addresses sparse rewards in software-engineering agents with guided trajectories and environment feedback. [Paper](https://arxiv.org/abs/2506.11425)

## 9.5 Hardware frontier

### Architecture is converging on data-movement economics

Current hardware directions include:

- HBM capacity/bandwidth growth;
- FP8/FP4 and microscaling support;
- larger and smarter on-chip SRAM use;
- high-bandwidth device-to-device fabrics;
- specialized engines for MoE routing or sparse operations;
- compiler-guided dataflow;
- phase-specialized serving hardware;
- efficient power and cooling at rack scale.

A good current exercise is to compare one workload, such as “70B model, 32K prompt, 2K output, 50 concurrent agent sessions,” across GPU, TPU, and LPU-style designs. You will quickly discover that no single headline metric answers the question.

---

# Part X — Engineering projects and capability milestones

## Project 1 — Build a tokenizer explorer

**Goal:** Make tokenization visible.

- Compare BPE / SentencePiece token counts on English, Hindi, code, JSON, and financial text.
- Calculate cost/context consequences.
- Build a visualization of token boundaries.

**You learn:** vocabulary, byte encodings, context economics.

## Project 2 — Implement a tiny GPT from first principles

**Goal:** Train a 1–20M parameter decoder model on a tiny dataset.

Implement:

- token embedding;
- causal mask;
- one attention head, then multi-head attention;
- residual connection;
- RMSNorm or LayerNorm;
- MLP;
- cross-entropy loss;
- sampling with temperature/top-p.

**You learn:** every equation in Part II becomes executable.

Resources:

- Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT)
- Karpathy’s [llm.c](https://github.com/karpathy/llm.c)
- [minGPT](https://github.com/karpathy/minGPT)

## Project 3 — Implement a KV cache and measure prefill versus decode

**Goal:** Build the physical intuition of generation.

- Generate with and without a KV cache.
- Record elapsed time for prefill and each decode token.
- Increase context length and plot memory/latency.
- Replace MHA with GQA in a toy implementation and compare cache size.

**You learn:** why architecture touches inference cost.

## Project 4 — Profile attention and implement tiled attention intuition

**Goal:** Understand FlashAttention’s motivation.

- Profile naïve attention.
- Measure memory allocation for \(T\times T\) score matrices.
- Write a blocked/tiled conceptual version.
- Read FlashAttention source or a Triton teaching implementation.

**You learn:** IO-aware algorithms and memory hierarchy.

## Project 5 — Serve a model with vLLM and SGLang

**Goal:** Treat serving as a systems problem.

- Deploy the same open model with both runtimes.
- Measure TTFT, TPOT, throughput, cache hit rate, and memory use.
- Test long prompts, shared prefixes, structured outputs, and concurrent requests.

**You learn:** schedulers, cache behavior, runtime differences, benchmark discipline.

## Project 6 — Quantization laboratory

**Goal:** Learn that bits are not a performance guarantee.

- Run BF16/FP16 baseline.
- Compare 8-bit and 4-bit weight quantization.
- Compare GPTQ/AWQ where runtime support exists.
- Test long-context behavior and structured-output validity.
- Record speed, memory, and quality in one report.

**You learn:** calibration, quality regression, kernel-dependent speedups.

## Project 7 — LoRA / QLoRA fine-tuning

**Goal:** Fine-tune an open model for a narrow task.

- Compare full fine-tune if feasible, LoRA, and QLoRA.
- Track memory, trainable parameters, validation quality, and serving impact.

**You learn:** low-rank adaptation and training-memory economics.

## Project 8 — Build a toy MoE layer

**Goal:** Feel the routing problem.

- Implement top-1 and top-2 routing.
- Log expert load histogram.
- Introduce synthetic routing imbalance.
- Add a balancing penalty or bias update.
- Compare capacity versus straggler behavior.

**You learn:** conditional computation and distributed-system implications.

## Project 9 — Implement an RLVR toy system

**Goal:** Understand RL around a verifier.

- Use arithmetic or unit-test code tasks.
- Sample multiple trajectories per prompt.
- Build a rule-based verifier.
- Implement group-relative normalized reward.
- Monitor reward hacking, answer length, entropy, and held-out tasks.

**You learn:** why the verifier is part of the learning algorithm.

## Project 10 — Capstone: an inference-aware agent platform

Build an agentic application that combines both curricula:

```text
agent harness
  → model router
  → vLLM/SGLang runtime
  → prefix caching
  → structured tool calls
  → model/quantization policy
  → latency/cost telemetry
  → verifier and eval suite
```

The capstone question is not “Does it answer?” It is:

> How does answer quality change as you trade model size, quantization, context length, cache policy, reasoning budget, and verifier strictness against latency and cost?

---

# Part XI — A 24-week learning path

## Phase 1 — Mathematical and neural-network foundations (Weeks 1–4)

**Week 1:** vectors, matrices, shapes, dot products, matrix multiplication. Build a NumPy linear layer.

**Week 2:** gradients, chain rule, backpropagation. Implement micrograd-style autodiff.

**Week 3:** probability, softmax, cross-entropy, sampling, entropy, KL divergence. Implement a categorical language-model loss.

**Week 4:** SGD, AdamW, normalization, mixed precision, numerical stability. Train a small MLP and intentionally break/fix it with learning-rate and precision changes.

## Phase 2 — GPT and Transformer engineering (Weeks 5–8)

**Week 5:** tokenizers, embeddings, causal language modeling. Build the tokenizer explorer.

**Week 6:** attention, masking, multi-head attention, positional representation. Implement a one-layer causal Transformer.

**Week 7:** residuals, RMSNorm, SwiGLU, GQA, KV cache. Build a tiny GPT.

**Week 8:** training loop, checkpoints, evaluation, sampling. Train and analyze a tiny model end to end.

## Phase 3 — Architecture alternatives and scaling (Weeks 9–11)

**Week 9:** MoE: routing, experts, balancing, expert parallelism. Build toy MoE.

**Week 10:** SSMs, Mamba, Mamba-2, hybrid models. Read Mamba and Jamba.

**Week 11:** linear attention, MLA, long-context architectures. Read Kimi Linear critically and write a comparison note: full attention vs GQA vs MLA vs SSM state.

## Phase 4 — Hardware and distributed training (Weeks 12–15)

**Week 12:** CPU/GPU memory hierarchy, kernels, roofline model, Tensor Cores. Read CUDA programming-model sections.

**Week 13:** TPU/LPU/ASIC comparison. Build a workload-specific hardware evaluation matrix.

**Week 14:** DDP, FSDP/ZeRO, tensor/pipeline/context/expert parallelism. Run a small distributed training experiment.

**Week 15:** data pipeline, scaling laws, mixed precision, checkpointing, failure recovery.

## Phase 5 — Inference systems (Weeks 16–19)

**Week 16:** prefill/decode and KV-cache math. Implement cache benchmark.

**Week 17:** FlashAttention, continuous batching, PagedAttention, vLLM.

**Week 18:** SGLang, prefix caching, structured outputs, multi-GPU serving.

**Week 19:** speculative decoding and prefill/decode disaggregation. Benchmark one model under realistic mixed-length load.

## Phase 6 — Quantization and post-training (Weeks 20–22)

**Week 20:** PTQ, GPTQ, AWQ, SmoothQuant, weights/activations/KV cache.

**Week 21:** LoRA, QLoRA, NF4, adapters, fine-tuning systems.

**Week 22:** RL basics, RLHF, PPO, DPO, RLVR, GRPO. Build toy verifier-based RL loop.

## Phase 7 — Research synthesis and capstone (Weeks 23–24)

**Week 23:** Read the frontier papers in Part IX. Write a “bottleneck transfer” memo for each: What cost moved? What new failure mode emerged?

**Week 24:** Build the inference-aware agent platform capstone. Publish a reproducible benchmark report with model/version, prompt mix, hardware, precision, runtime, cache policy, batch policy, quality metrics, and cost estimates.

---

# Part XII — Canonical references and live sources

## Mathematics and deep learning

1. [Mathematics for Machine Learning](https://mml-book.github.io/)
2. [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
3. *Deep Learning* — Goodfellow, Bengio, Courville. https://www.deeplearningbook.org/
4. *Understanding Deep Learning* — Simon J. D. Prince. https://udlbook.github.io/udlbook/
5. [micrograd](https://github.com/karpathy/micrograd)
6. [Dive into Deep Learning](https://d2l.ai/)

## Core model architecture

7. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
8. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
9. [An Introduction to Transformers](https://arxiv.org/abs/2304.10557)
10. [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
11. [RMSNorm](https://arxiv.org/abs/1910.07467)
12. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
13. [GQA](https://arxiv.org/abs/2305.13245)

## MoE and alternative architectures

14. [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)
15. [GShard](https://arxiv.org/abs/2006.16668)
16. [Switch Transformers](https://arxiv.org/abs/2101.03961)
17. [Mixtral](https://arxiv.org/abs/2401.04088)
18. [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
19. [Mamba](https://arxiv.org/abs/2312.00752)
20. [Mamba-2 / State Space Duality](https://arxiv.org/abs/2405.21060)
21. [Jamba](https://arxiv.org/abs/2403.19887)
22. [Kimi Linear](https://arxiv.org/abs/2510.26692)

## Training and distributed systems

23. [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
24. [Megatron-LM at scale](https://arxiv.org/abs/2104.04473)
25. [ZeRO](https://arxiv.org/abs/1910.02054)
26. [PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html)
27. [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
28. [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)

## Hardware and programming systems

29. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
30. [Google TPU architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
31. [Groq LPU architecture](https://groq.com/lpu-architecture)
32. [AWS Neuron](https://aws.amazon.com/ai/machine-learning/neuron/)
33. [AMD MI300 architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
34. [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)

## Inference and serving

35. [FlashAttention](https://arxiv.org/abs/2205.14135)
36. [FlashAttention-2](https://arxiv.org/abs/2307.08691)
37. [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
38. [vLLM repository](https://github.com/vllm-project/vllm)
39. [SGLang](https://github.com/sgl-project/sglang)
40. [SGLang docs](https://docs.sglang.ai/)
41. [Speculative Decoding](https://arxiv.org/abs/2211.17192)
42. [Medusa](https://arxiv.org/abs/2401.10774)
43. [EAGLE](https://arxiv.org/abs/2401.15077)
44. [EAGLE-3](https://arxiv.org/abs/2503.01840)
45. [Efficient LLM inference survey](https://arxiv.org/abs/2404.14294)

## Quantization and efficient adaptation

46. [GPTQ](https://arxiv.org/abs/2210.17323)
47. [SmoothQuant](https://arxiv.org/abs/2211.10438)
48. [AWQ](https://arxiv.org/abs/2306.00978)
49. [QLoRA](https://arxiv.org/abs/2305.14314)
50. [Low-bit LLM survey](https://arxiv.org/abs/2409.16694)
51. [Quantized LLM evaluation](https://arxiv.org/abs/2402.18158)

## RL and post-training

52. [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155)
53. [PPO](https://arxiv.org/abs/1707.06347)
54. [DPO](https://arxiv.org/abs/2305.18290)
55. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
56. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
57. [DAPO](https://arxiv.org/abs/2503.14476)
58. [Agent-RLVR](https://arxiv.org/abs/2506.11425)
59. [RLVR and correct reasoning analysis](https://arxiv.org/abs/2506.14245)
60. [Spurious Rewards](https://arxiv.org/abs/2506.10947)

---

# A note on currency

This guide was assembled with sources checked through **June 23, 2026**. Architecture and math fundamentals change slowly; model releases, hardware specifications, runtime APIs, benchmark numbers, and vendor support matrices change quickly.

Before relying on a fast-moving claim, check:

- the paper’s latest version and released code;
- the model card / technical report;
- runtime release notes for vLLM, SGLang, TensorRT-LLM, Neuron, or the relevant stack;
- accelerator vendor documentation for exact precision, memory, and interconnect support;
- benchmark methodology: model version, prompt/output lengths, batch size, hardware, runtime, precision, and metric definition.

> **Final engineering rule:** Treat every claimed speedup as a conditional statement. Ask: *on which model, on which hardware, at which precision, under which workload, compared with which baseline, while preserving which quality metric?*

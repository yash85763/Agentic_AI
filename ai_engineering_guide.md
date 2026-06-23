# AI Engineering
## A Research-Grade, Systems-First Guide to LLM Mathematics, Architectures, Hardware, Training, Inference, Quantization, and Reinforcement Learning

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
>
> **Research-grade learning contract:** This is not a feed of the newest model releases. Every subject in this guide should be learned in five passes: **(1) the original problem and historical lineage, (2) the mathematical object and assumptions, (3) the canonical algorithm, (4) the modern implementation and systems trade-offs, and (5) the active research disputes.** A paper is not “learned” when you can repeat its headline; it is learned when you can derive its objective, reproduce a constrained version, identify its hidden assumptions, and design an experiment that could falsify its claim.

---

# Table of Contents

- [Part 0 — The map: what an AI engineer actually engineers](#part-0--the-map-what-an-ai-engineer-actually-engineers)
- [How to use this guide at research depth](#how-to-use-this-guide-at-research-depth)
- [Part I — Mathematics you need before models](#part-i--mathematics-you-need-before-models)
- [Part I-A — Mathematics research track: a bounded 12-month curriculum](#part-i-a--mathematics-research-track-a-bounded-12-month-curriculum)
- [Part II — From text to GPT-style language models](#part-ii--from-text-to-gpt-style-language-models)
- [Part III — Architecture families beyond the dense Transformer](#part-iii--architecture-families-beyond-the-dense-transformer)
- [Part IV — Training foundation models](#part-iv--training-foundation-models)
- [Part V — The physical substrate: CPU, GPU, TPU, LPU, memory, and networks](#part-v--the-physical-substrate-cpu-gpu-tpu-lpu-memory-and-networks)
- [Part VI — Inference and serving systems](#part-vi--inference-and-serving-systems)
- [Part VII — Quantization, compression, and efficient adaptation](#part-vii--quantization-compression-and-efficient-adaptation)
- [Part VIII — Reinforcement learning and post-training](#part-viii--reinforcement-learning-and-post-training)
- [Part VIII-A — RL implementation laboratory and research agenda](#part-viii-a--rl-implementation-laboratory-and-research-agenda)
- [Part IX — Current research frontier, 2025–June 2026](#part-ix--current-research-frontier-2025june-2026)
- [Part X — Engineering projects and capability milestones](#part-x--engineering-projects-and-capability-milestones)
- [Part XI — A 24-week engineering foundation](#part-xi--a-24-week-engineering-foundation)
- [Part XI-A — A 12-month PhD-style research path](#part-xi-a--a-12-month-phd-style-research-path)
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

## How to use this guide at research depth

This curriculum supports two complementary modes. Neither is optional if you want to become an AI engineering researcher rather than merely an API integrator.

### Engineering mode: make the system real

For every topic, ask:

- Can I implement the minimal version?
- Can I profile where time and memory go?
- Can I build an evaluation that detects regressions?
- Can I explain the production failure mode?

### Research mode: understand the claim, not the headline

For every canonical paper, produce a one-page **research card**:

```text
Problem: What failed in the previous approach?
Formal object: What distribution, objective, state, or computational bottleneck is being changed?
Assumptions: What must be true for the method to work?
Mechanism: Which term in the equation or system is responsible for the claimed gain?
Evidence: Which benchmark, baseline, budget, hardware, and metric support the claim?
Ablations: What experiment separates the mechanism from confounders?
Failure modes: When should the method not be used?
Replication: What is the smallest faithful experiment I can run?
Open question: What would I test next?
```

### The five-pass method

1. **Archaeology:** Learn the original problem and why a previous solution broke.
2. **Derivation:** Derive the central objective or computation on paper, including shapes and units.
3. **Toy implementation:** Reproduce the mechanism on a small environment where you can inspect every tensor and every failure.
4. **Systems implementation:** Understand the distributed, numerical, memory, kernel, and data-pipeline compromises required at real scale.
5. **Research critique:** Read current papers as conditional claims. Identify what moved, what was held fixed, what the benchmark cannot prove, and what would falsify the conclusion.

### A warning about chronological learning

Chronology is necessary but insufficient. You should not study PPO merely because it came before DPO, nor GRPO merely because it is newer. Learn them as answers to different constraints:

```text
control / preference data available?        → preference modeling or direct preference optimization
cheap deterministic correctness check?      → verifier-based RL / RLVR
long multi-step environment?                → agent RL, curriculum, environment design
need stable offline training?               → DPO-family / offline regularized objectives
need on-policy exploration or reward gains? → PPO/GRPO-family online RL
need scalable deployed behavior feedback?   → online or asynchronous post-training, with strict evaluation
```

The rest of the guide uses this stance. Part VIII is deliberately more detailed because reinforcement learning is where mathematical assumptions, data quality, evaluation, inference throughput, distributed systems, and alignment questions collide most visibly.

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

# Part I-A — Mathematics Research Track: A Bounded 12-Month Curriculum

> **Purpose:** This is the mathematics-only companion to the AI Engineering Guide. It answers a question that ordinary curricula often avoid: *how much mathematics is enough before I move on?* The answer is not “all of it.” Mathematics is an ocean with no final shoreline. The aim is **working mathematical fluency for AI engineering**, followed by a visible map of the deeper territory you can enter when a paper requires it.
>
> **Use this track in one of two ways:**
>
> 1. **Dedicated mode:** 10–14 hours per week for 12 months before taking on a research-heavy model-systems project.
> 2. **Parallel mode:** 6–8 hours per week alongside implementation. In this mode, double the calendar time, but keep the order unchanged.
>
> Do not treat this as a gatekeeping ritual. A matrix only becomes alive when it becomes an embedding projection, a Jacobian when it becomes backpropagation, a Markov chain when it becomes a policy rollout, and a condition number when it explains why a low-precision kernel failed.

## I-A.1 What “enough mathematics” means

There are three levels of knowing a mathematical topic.

| Level | What you can do | What it is sufficient for |
|---|---|---|
| **Operational fluency** | Read notation, check shapes/units, use a library correctly, explain the idea qualitatively | Build and debug standard AI systems |
| **Derivational fluency** | Derive the objective, gradients, update rule, and complexity of a method without copying a tutorial | Read most architecture/training/inference papers critically |
| **Research fluency** | State assumptions, prove small lemmas, construct counterexamples, analyze failure modes, and modify the method | Formulate and defend original research |

The twelve-month core gets you to **derivational fluency** in the mathematics that appears repeatedly across modern AI engineering. The **C extensions** below are the extra depth that moves you toward research fluency in a chosen subfield.

### The A–B–C convention used in every month

- **A — Anchor concepts:** non-negotiable concepts. You should be able to use them without looking them up.
- **B — Bridge concepts:** the exact bridge from the mathematics to model architectures, optimization, systems, or reinforcement learning.
- **C — Creative/research extension:** additional concepts to study when you want to write research, reproduce a technical paper, or pursue a specific frontier.

This is deliberately asymmetric. You must master **A**, connect it through **B**, and selectively go deeper through **C**. Reading every C topic immediately is the fastest way to turn a good plan into a beautiful abandoned notebook.

### The hard stop rule

Move to the next month when you can do all four:

1. **Define** every Anchor concept in your own words and state its assumptions.
2. **Derive** the month’s listed equations on paper, including shapes.
3. **Implement** one small numerical experiment without copying a full solution.
4. **Diagnose** one failure or counterexample, not merely one success case.

You do **not** need to complete every exercise in every book. You do need enough problems to expose whether your understanding survives contact with algebra.

---

## I-A.2 The reference shelf: use a few books deeply, not twenty books shallowly

The guide uses a small set of **spine texts**. Their chapter numbers are stable in the linked editions, but use chapter *titles* as the final source of truth because page numbers and printings vary.

| Short name | Role in the curriculum | Core chapters / sections | Research extension |
|---|---|---|---|
| **MML** — Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* | The main bridge text that keeps the mathematics connected to ML | Ch. 2–7: Linear Algebra, Analytic Geometry, Matrix Decompositions, Vector Calculus, Probability and Distributions, Continuous Optimization | Ch. 8–10 for data geometry, regression, PCA; later ML chapters when needed |
| **DLB** — Goodfellow, Bengio, Courville, *Deep Learning* | The compact technical reference for how mathematics appears in deep learning | Ch. 2–4 and Ch. 8: Linear Algebra; Probability & Information Theory; Numerical Computation; Optimization | Ch. 6, 9, 10, 11 for networks, CNNs, sequence models, practical methodology |
| **UDL** — Simon J. D. Prince, *Understanding Deep Learning* | A modern explanatory companion with matrix-calculus and probability appendices | Math-for-ML and linear-algebra foundations plus Appendix B (Matrix Calculus) and Appendix C (Probability) | Chapters on optimization, transformers, generative models, and RL as you reach them |
| **B&H** — Blitzstein and Hwang, *Introduction to Probability* | Probability intuition and disciplined problem solving | Ch. 1–7: probability, conditioning, random variables, continuous variables, moments, joint distributions, conditional expectation | Later chapters on transforms, convergence, and stochastic processes |
| **CT** — Cover and Thomas, *Elements of Information Theory* | The canonical source for entropy, KL, mutual information, and coding ideas | Ch. 2 first: entropy, relative entropy, mutual information | Channels, rate–distortion, information geometry only when your research needs them |
| **BV** — Boyd and Vandenberghe, *Convex Optimization* | Convexity, duality, and disciplined optimization reasoning | Ch. 2–5: convex sets, functions, problems, duality; then Ch. 9 | Ch. 10–11 for constrained minimization and interior-point methods |
| **TB** — Trefethen and Bau, *Numerical Linear Algebra* | Conditioning, stability, SVD, least squares, iterative computation | Lectures 1–5 first: matrix-vector multiplication, orthogonality, norms, SVD | Least squares, Krylov methods, eigenvalue algorithms, preconditioning |
| **H** — Higham, *Accuracy and Stability of Numerical Algorithms* | Why finite-precision computation changes the algorithm you think you are running | Ch. 1 on floating-point arithmetic; selected sections on rounding/error analysis | Deeper backward-error analysis and matrix-factorization stability |
| **SB** — Sutton and Barto, *Reinforcement Learning: An Introduction* | The mathematical spine for sequential decision-making and policy gradients | Ch. 2–6, then Ch. 13 | Ch. 7–12 for approximation, eligibility traces, off-policy prediction/control |

### The “six chapters out of ten” rule

When you open a serious book, do not read it linearly by default. Use this pattern:

```text
Core pass:       the 50–60% of chapters that define the reusable language
Application pass: selected chapters that connect the language to your AI question
Research pass:    only the chapters whose assumptions appear in the papers you are reading
```

For example, **MML Chapters 2–7** cover the reusable mathematics core. You do not need to finish its later machine-learning chapters before beginning Transformers. Likewise, **DLB Chapters 2–4 and 8** explain the mathematics that will recur everywhere; the other chapters are application bridges, not a prerequisite wall. The Deep Learning book’s official table of contents identifies Chapters 2–4 as linear algebra, probability/information theory, and numerical computation, and Chapter 8 as optimization. [DLB table of contents](https://www.deeplearningbook.org/contents/TOC.html)  
MML’s first mathematical-foundations block is Chapters 2–7. [MML chapter map](https://www.cambridge.org/highereducation/books/mathematics-for-machine-learning/5EE57FD1CFB23E6EB11E130309C7EF98/vector-calculus/0C7DF518C4CBA7F79E739FC41C444197)

---

## I-A.3 Monthly calendar at a glance

| Month | Mathematical theme | AI engineering questions it unlocks |
|---|---|---|
| 1 | Mathematical language + linear algebra I | What is a representation? Why does matrix multiplication transform features? |
| 2 | Linear algebra II + geometry + decompositions | Why do SVD, PCA, LoRA, norms, and conditioning matter? |
| 3 | Multivariable and matrix calculus | How does backpropagation work through matrices, softmax, and attention? |
| 4 | Probability foundations | What exactly is a model distribution, expectation, likelihood, and sampling process? |
| 5 | Statistics, estimation, and uncertainty | Why do datasets, estimators, generalization, and confidence claims behave as they do? |
| 6 | Information theory | Why are cross-entropy, KL, perplexity, mutual information, and coding central to LLMs? |
| 7 | Convex optimization | What does it mean to minimize an objective, constrain a model, or reason with duality? |
| 8 | Stochastic and deep optimization | Why do minibatches, AdamW, schedules, clipping, and noise shape training? |
| 9 | Numerical linear algebra + floating point | Why do BF16/FP8/quantization and kernels produce different results from the equations? |
| 10 | Markov processes + dynamic programming | What is an MDP, value function, Bellman equation, and occupancy distribution? |
| 11 | Policy gradients and regularized RL | How do PPO, DPO, RLHF, RLVR, KL control, and advantage estimation arise mathematically? |
| 12 | Dynamical systems, generalization, and research synthesis | How do SSM/Mamba, stability, scaling claims, and research assumptions fit into one mathematical map? |

### Standard four-week rhythm within every month

| Week | Work | Evidence you produce |
|---|---|---|
| **1 — Objects** | Definitions, notation, geometry, basic exercises | One-page glossary and shape/assumption sheet |
| **2 — Derivations** | Work through the central equations by hand | A derivation notebook, with every intermediate shape |
| **3 — Problems** | Closed-book exercises and counterexamples | Corrected problem set and error log |
| **4 — Computation + paper bridge** | Minimal NumPy/PyTorch experiment; read one canonical paper section | Small repo/notebook plus a research card |

Keep one **math failure log**. Every time you make a mistake, record the category: shape mismatch, transpose error, conditioning assumption, probability conditioning error, sign error, invalid interchange of expectation/gradient, or numerical overflow. That log becomes a very personal atlas of the cliffs you are most likely to fall off later.

---

# Month 1 — Mathematical language and linear algebra I

> **Question:** What kind of object is a model state, and what does a neural-network layer actually do to it?

### A — Anchor concepts

- Sets, functions, relations, domains/codomains, composition, inverse images.
- Scalars, vectors, matrices, tensors; index notation and shape discipline.
- Vector addition, scalar multiplication, span, linear independence, basis, dimension.
- Matrix-vector multiplication as a **linear map**.
- Matrix-matrix multiplication as composition of linear maps.
- Dot product, Euclidean norm, cosine similarity, orthogonality, projection.
- Transpose, identity, inverse where it exists, diagonal matrices.

### B — Bridge concepts to AI

- Embedding lookup is a row selection from an embedding matrix, equivalent mathematically to a one-hot vector multiplied by that matrix.
- A dense layer is an affine map: 

  \[
  h = xW + b.
  \]

- The query, key, and value maps in attention are learned linear transformations of the same hidden representation.
- Tensor shapes are part of the proof. If 

  \[
  X \in \mathbb{R}^{B\times T\times d}, \quad W_Q \in \mathbb{R}^{d\times d_h},
  \]

  then 

  \[
  Q=XW_Q \in \mathbb{R}^{B\times T\times d_h}.
  \]

  If you cannot state the shape, you do not yet know what operation is occurring.

### C — Creative/research extension

- Abstract vector spaces and dual spaces.
- Linear operators, null spaces, column/row spaces, rank-nullity theorem.
- Bilinear forms and inner-product spaces.
- Tensor notation and Einstein summation (`einsum`) for attention and distributed tensor layouts.

### Read this, not everything

- **MML:** Chapter 2, all sections.
- **DLB:** Chapter 2, especially sections on vectors/matrices/tensors, multiplication, identity/inverse, linear dependence/independence, norms, eigendecomposition, and SVD preview.
- **Strang, *Introduction to Linear Algebra*:** chapters on elimination, vector spaces, and linear transformations, selected by title.
- **3Blue1Brown:** the first five *Essence of Linear Algebra* videos for visual intuition, then immediately return to written exercises.

### Derivation and proof targets

1. Prove that matrix multiplication represents composition: 
   \[
   (xA)B=x(AB).
   \]
2. Show that the dot product can be written as 
   \[
   x^Ty.
   \]
3. Derive the projection of a vector \(x\) onto a nonzero vector \(u\): 
   \[
   \operatorname{proj}_u(x)=\frac{u^Tx}{u^Tu}u.
   \]
4. Given a two-layer linear network \(y=xW_1W_2\), explain why it is still only one linear map without a nonlinearity.

### Computation laboratory

- Implement matrix multiplication with three nested loops, then compare it to NumPy/PyTorch.
- Generate two-dimensional vectors; project them onto a line and visualize the residual as an orthogonal error.
- Implement a toy embedding table and show that one-hot multiplication and index lookup return the same vector.
- Write a `check_shape()` helper that validates dimensions for a toy attention projection.

### Stop rule

You are ready to move on when you can inspect an equation such as 
\(Q=XW_Q\), \(K=XW_K\), \(V=XW_V\) and explain, without handwaving, what space each object lives in and why each multiplication is legal.

### When to return here later

Return whenever a paper mentions **representation space**, **projection**, **rank**, **hidden width**, **head dimension**, **tensor parallel sharding**, or a tensor shape you cannot parse.

---

# Month 2 — Geometry, spectral ideas, and matrix decompositions

> **Question:** Which directions in a high-dimensional space matter, how do matrices stretch or rotate them, and when can a large transformation be approximated by a smaller one?

### A — Anchor concepts

- Eigenvalues and eigenvectors; diagonalization and when it fails.
- Symmetric matrices, positive semidefinite (PSD) matrices, quadratic forms.
- Singular value decomposition (SVD): 
  \[
  W=U\Sigma V^T.
  \]
- Rank, null space, column space, row space.
- Orthogonal matrices; orthonormal bases.
- Least squares and normal equations.
- Condition number and the distinction between a mathematically solvable problem and a numerically well-behaved one.

### B — Bridge concepts to AI

- **PCA** uses eigenvectors/SVD to identify high-variance directions.
- **LoRA** relies on the empirical fact that useful parameter updates can often be represented with low rank: 
  \[
  \Delta W \approx BA, \quad r\ll\min(d_{in},d_{out}).
  \]
- Attention logits are dot products in learned geometry; scaling by \(1/\sqrt{d_h}\) controls their typical magnitude.
- PSD matrices appear in covariance matrices, Hessian approximations, kernel methods, and Fisher-information-related ideas.
- Conditioning predicts sensitivity. A nearly singular matrix may amplify small numerical or data perturbations into large solution changes.

### C — Creative/research extension

- Spectral graph theory, Laplacians, and graph neural networks.
- Random matrix theory for initialization, spectra, and scaling laws.
- Matrix perturbation theory: Weyl inequalities, Davis–Kahan theorem.
- Tensor decompositions for multimodal models and compression.

### Read this, not everything

- **MML:** Chapters 3 and 4.
- **DLB:** finish the eigenvalue/SVD material in Chapter 2.
- **TB:** Lectures 1–5, which introduce matrix-vector multiplication, orthogonality, norms, and SVD. [Trefethen & Bau overview](https://people.maths.ox.ac.uk/trefethen/text.html)
- **UDL:** its linear-algebra chapter and Appendix B as a future reference for derivative identities. The book’s online site lists linear algebra topics including linear transformations, null spaces, orthogonal matrices, and SVD. [UDL](https://udlbook.github.io/udlbook/)

### Derivation and proof targets

1. Derive least squares by minimizing 
   \[
   \min_w \|Xw-y\|_2^2
   \]
   and obtaining the normal equations \(X^TXw=X^Ty\).
2. Show why \(X^TX\) is PSD: 
   \[
   z^TX^TXz=\|Xz\|_2^2\ge 0.
   \]
3. Explain the geometric meaning of each factor in SVD: \(V^T\) rotates/reflections input coordinates, \(\Sigma\) scales orthogonal directions, \(U\) rotates/reflections output coordinates.
4. Derive the best rank-\(r\) approximation intuition from truncated SVD, then verify it numerically.

### Computation laboratory

- Build PCA from SVD on a small image or embedding dataset. Compare reconstruction error for ranks \(r=1,2,\ldots\).
- Implement a low-rank update \(BA\) to a frozen matrix and compare parameter count against a full \(\Delta W\).
- Create ill-conditioned linear systems and observe how tiny perturbations change the solution.

### Stop rule

You are ready when you can read “we use a rank-\(r\) adapter,” “the covariance is PSD,” or “the method suffers from poor conditioning” and turn each phrase into an equation, a geometry picture, and a small test.

### When to return here later

Return for **LoRA**, **quantization error analysis**, **PCA**, **spectral initialization**, **Hessian/Fisher approximations**, **MoE routing projections**, and **tensor parallelism**.

---

# Month 3 — Multivariable calculus, matrix calculus, and automatic differentiation

> **Question:** How can a scalar loss send a useful learning signal backward through billions of coupled parameters without constructing impossible-sized Jacobians?

### A — Anchor concepts

- Limits, continuity, differentiability, directional derivatives.
- Partial derivatives, gradient, gradient direction, tangent-plane approximation.
- Jacobian of a vector-valued function; Hessian of a scalar-valued function.
- Chain rule in scalar, vector, and matrix form.
- Vector-Jacobian products (VJPs) and Jacobian-vector products (JVPs).
- Taylor expansion through second order.
- Gradients of affine maps, elementwise nonlinearities, dot products, norms, softmax, log-softmax, and cross-entropy.

### B — Bridge concepts to AI

- Backpropagation is reverse-mode automatic differentiation: it computes VJPs, not full Jacobians.
- In a linear layer \(h=xW+b\), you need gradients with respect to \(x\), \(W\), and \(b\).
- The crucial softmax-cross-entropy identity is: for logits \(z\), target one-hot vector \(y\), and \(p=\operatorname{softmax}(z)\),
  \[
  \frac{\partial\mathcal{L}}{\partial z}=p-y.
  \]
  This compact result is one of the hinges of language-model training.
- Attention is differentiable composition: projections \(\rightarrow\) scores \(\rightarrow\) softmax \(\rightarrow\) weighted sum. Its backward pass is merely the chain rule at scale.

### C — Creative/research extension

- Fréchet derivatives and differential notation.
- Hessian-vector products, Gauss–Newton approximations, Fisher information.
- Implicit differentiation through an optimizer or equilibrium layer.
- Differential geometry of parameter spaces and natural gradient.

### Read this, not everything

- **MML:** Chapter 5.
- **DLB:** Chapter 4 for numerical computation and differentiation context; keep Chapter 2’s derivative identities nearby.
- **UDL:** Appendix B, *Matrix Calculus*, as the main reference sheet.
- **micrograd:** read the code slowly enough to trace each local derivative. [micrograd](https://github.com/karpathy/micrograd)
- **PyTorch:** Autograd mechanics after you can manually derive a small graph. [PyTorch Autograd](https://pytorch.org/docs/stable/notes/autograd.html)

### Derivation and proof targets

1. Derive all three gradients for 
   \[
   L=\frac12\|xW+b-y\|_2^2.
   \]
2. Derive 
   \[
   \nabla_x(x^TAx)=(A+A^T)x,
   \]
   and simplify when \(A\) is symmetric.
3. Derive \(\partial\mathcal{L}/\partial z=p-y\) for softmax cross-entropy.
4. Write the local derivatives for a two-layer MLP and recover the full gradient by the chain rule.
5. Explain, in words and shapes, why reverse-mode AD is cheap for a scalar loss with many parameters.

### Computation laboratory

- Implement scalar reverse-mode autodiff in the style of `micrograd`.
- Implement a two-layer MLP with NumPy and manual backprop; check every gradient with finite differences.
- Deliberately introduce an incorrect transpose in the backward pass and use numerical gradient checks to locate it.
- Implement a single-head attention forward pass; derive and test gradients for a tiny \(T=3, d_h=2\) case.

### Stop rule

You are ready when you can derive a training update for a small model from the loss, and when you understand why autograd sometimes fails: in-place mutation, detached tensors, unstable operations, or an incorrect computational graph.

### When to return here later

Return for **new loss functions**, **RL policy gradients**, **custom CUDA/Triton kernels with backward passes**, **implicit layers**, **second-order methods**, **normalization layers**, and **attention variants**.

---

# Month 4 — Probability foundations: distributions, expectation, and conditioning

> **Question:** What does it mean for a language model to assign probability to text, and why does sampling from that distribution differ from predicting a single “answer”? 

### A — Anchor concepts

- Sample spaces, events, sigma-algebra intuition only, probability axioms.
- Conditional probability and Bayes’ rule.
- Discrete and continuous random variables; PMF, PDF, CDF.
- Expectation, variance, covariance, correlation.
- Joint, marginal, and conditional distributions.
- Law of total expectation and law of total variance.
- Independence and conditional independence.
- Common distributions: Bernoulli, categorical, binomial, Gaussian, multinomial, geometric, Poisson.

### B — Bridge concepts to AI

- A decoder LM defines 
  \[
  p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
  \]
  This is repeated conditional probability, not a single global “truth score.”
- Sampling temperature modifies a categorical distribution derived from logits; it does not change the trained weights.
- Minibatch loss is a random estimator of a population objective.
- Dropout, stochastic decoding, RL rollouts, and randomized routing all require distinguishing an expectation from one sampled outcome.

### C — Creative/research extension

- Measure-theoretic probability: sigma-algebras, measurable functions, Lebesgue integration.
- Characteristic functions, moment-generating functions, concentration inequalities.
- Exchangeability, de Finetti-style thinking, and Bayesian nonparametrics.
- Probabilistic graphical models and conditional-independence structure.

### Read this, not everything

- **B&H:** Chapters 1–7, solved actively rather than passively.
- **MML:** Chapter 6 for ML-oriented notation.
- **DLB:** Chapter 3, especially probability rules, common distributions, expectation/variance/covariance, and information theory setup.
- **UDL:** Appendix C, *Probability*, as a compressed companion.
- **Murphy, PML: An Introduction:** use the early probability and inference chapters by topic when you want a second explanation. [PML](https://probml.github.io/pml-book/book1.html)

### Derivation and proof targets

1. Derive Bayes’ rule from the definition of conditional probability.
2. Prove the law of total expectation for a discrete conditioning variable.
3. Show that independent variables have zero covariance, then construct a counterexample showing that zero covariance does **not** imply independence.
4. Derive the mean and variance of a Bernoulli random variable.
5. Express next-token negative log-likelihood as an expectation over the data distribution.

### Computation laboratory

- Simulate a categorical next-token distribution and show how temperature, top-\(k\), and nucleus sampling affect samples and entropy.
- Use Monte Carlo estimation to approximate an expectation; graph error as sample count grows.
- Build a small Bayesian diagnostic example showing prior, likelihood, posterior, and posterior predictive distribution.

### Stop rule

You are ready when you can read an objective containing \(\mathbb{E}_{x\sim p}[f(x)]\), identify what is random, what distribution is being sampled, and what Monte Carlo estimator an implementation is using.

### When to return here later

Return for **language modeling**, **sampling**, **uncertainty estimates**, **Bayesian optimization**, **Monte Carlo RL**, **mixture-of-experts routing**, and **evaluation confidence intervals**.

---

# Month 5 — Statistics, estimation, and generalization

> **Question:** How do finite datasets justify claims about an unknown population, and which parts of a model result are signal versus experiment noise?

### A — Anchor concepts

- Population, sample, statistic, estimator, bias, variance, consistency.
- Likelihood, log-likelihood, maximum likelihood estimation (MLE), maximum a posteriori estimation (MAP).
- Train/validation/test separation and leakage.
- Sampling distributions, bootstrap intuition, confidence intervals.
- Hypothesis tests, Type I/II errors, effect size, power.
- Bias–variance trade-off and regularization.
- Empirical risk versus population risk.

### B — Bridge concepts to AI

- Cross-entropy training is MLE under a categorical model.
- Weight decay can be interpreted as a prior in some settings, but optimizer details matter; do not casually equate all L2 penalties with all forms of weight decay.
- Benchmark numbers without uncertainty or fixed evaluation protocols can create false progress.
- Scaling curves, A/B comparisons, ablations, and serving benchmarks are statistical arguments, whether or not their authors say so.

### C — Creative/research extension

- PAC learning, VC dimension, Rademacher complexity, uniform convergence.
- PAC-Bayes and compression-based generalization bounds.
- Causal inference: potential outcomes, confounding, interventions.
- Sequential testing and adaptive data analysis.

### Read this, not everything

- **MML:** Chapter 8, *When Models Meet Data*, then Chapter 9 for linear regression as a complete worked statistical model.
- **DLB:** Chapter 5 for ML basics and Chapter 7 for regularization (read selectively by title).
- **Murphy PML:** chapters on linear regression, logistic regression, regularization, and model selection by title.
- **Wasserman, *All of Statistics*:** the estimation, confidence interval, testing, and bootstrap chapters as a concise research extension.

### Derivation and proof targets

1. Derive MLE for a Bernoulli parameter and show it equals the sample mean.
2. Derive ridge regression’s objective and normal equations.
3. Show the bias–variance decomposition for squared error under its assumptions.
4. Explain why repeatedly tuning on a test set makes it cease to be a true test set.
5. Design a confidence interval or bootstrap procedure for a model-comparison metric.

### Computation laboratory

- Fit linear regression by closed-form least squares and by gradient descent; compare when conditioning is poor.
- Run a bootstrap confidence interval for a metric difference between two small models.
- Create a deliberately leaky split and quantify the apparent but false performance gain.

### Stop rule

You are ready when you can read “our method improves benchmark accuracy by 1.2 points” and immediately ask: compared on which frozen split, with how many seeds, what uncertainty, what tuning budget, and what was held constant?

### When to return here later

Return for **benchmark design**, **model scaling claims**, **ablation studies**, **evals**, **reward-model validation**, **A/B testing**, and **research-paper critique**.

---

# Month 6 — Information theory and the mathematics of language modeling

> **Question:** Why does next-token prediction use cross-entropy, why does KL divergence appear in alignment, and what does perplexity actually measure?

### A — Anchor concepts

- Self-information: \(-\log p(x)\).
- Entropy: \(H(P)\).
- Cross-entropy: \(H(P,Q)\).
- KL divergence: \(D_{KL}(P\|Q)\).
- Mutual information: \(I(X;Y)\).
- Chain rule of entropy.
- Jensen’s inequality and non-negativity of KL divergence.
- Perplexity and its dependence on tokenization and evaluation protocol.

### B — Bridge concepts to AI

- Minimizing token cross-entropy is minimizing an empirical estimate of 
  \[
  H(p_{data},p_\theta)=H(p_{data})+D_{KL}(p_{data}\|p_\theta).
  \]
  Since \(H(p_{data})\) does not depend on \(\theta\), training reduces forward KL from data to model.
- In RLHF/PPO-style post-training, a reverse-direction policy KL penalty often constrains deviation from a reference policy:
  \[
  D_{KL}(\pi_\theta\|\pi_{ref}).
  \]
  The direction matters.
- Tokenization changes entropy units and perplexity comparisons. Compare perplexity only when tokenization and data protocol are compatible.
- Mutual information provides language for discussing representation, retrieval, compression, and bottleneck claims, but it is often difficult to estimate reliably in deep networks.

### C — Creative/research extension

- Rate–distortion theory and lossy compression.
- Minimum description length (MDL) and model selection.
- Information bottleneck, variational bounds, contrastive objectives.
- Information geometry and Fisher-Rao metrics.

### Read this, not everything

- **CT:** Chapter 2 is mandatory. It introduces entropy, relative entropy, and mutual information.
- **DLB:** Chapter 3’s information-theory material.
- **MML:** revisit Chapter 6 where probabilistic notation becomes operational.
- **MacKay, *Information Theory, Inference, and Learning Algorithms*:** entropy and coding chapters for a more intuitive second pass.

### Derivation and proof targets

1. Derive cross-entropy decomposition into entropy plus KL divergence.
2. Prove \(D_{KL}(P\|Q)\ge 0\) using Jensen’s inequality at the level appropriate to your calculus/probability background.
3. Derive why the geometric mean of inverse token probabilities leads to perplexity.
4. Compare forward and reverse KL with a two-mode discrete example; show their different mode-covering/mode-seeking tendencies.

### Computation laboratory

- Compute entropy, cross-entropy, KL, and perplexity for hand-built categorical distributions.
- Show how a tokenization change can alter perplexity numerically even when the underlying text is the same.
- Implement a KL-regularized categorical policy update in a tiny bandit.

### Stop rule

You are ready when you can see a loss containing cross-entropy or KL and explain the two distributions, the direction of the divergence, the implied behavior, and the metric’s limitations.

### When to return here later

Return for **pretraining objectives**, **DPO/RLHF**, **distillation**, **variational methods**, **compression**, **retrieval objectives**, **privacy**, and **calibration**.

---

# Month 7 — Convex optimization: the clean world that teaches you how optimization arguments work

> **Question:** What can be guaranteed when an objective has good geometry, and why is deep-learning optimization harder precisely because those guarantees disappear?

### A — Anchor concepts

- Convex sets, convex functions, strictly/strongly convex functions.
- First-order condition for convexity.
- Convex optimization problems, feasible sets, constraints.
- Lagrangian, primal/dual problems, weak/strong duality, KKT conditions.
- Gradient descent and projected gradient descent.
- Smoothness, Lipschitz gradients, strong convexity, convergence intuition.

### B — Bridge concepts to AI

- Linear regression, logistic regression with standard regularization, SVMs, and many calibration subproblems are convex.
- Convexity is a **control experiment for intuition**. Neural networks are nonconvex, but the vocabulary of objectives, constraints, dual variables, and conditioning remains useful.
- Duality appears in constrained optimization, distributionally robust optimization, optimal transport, and some preference-learning formulations.
- Trust-region and KL-constrained RL methods borrow the language of constrained optimization even when the final problem is nonconvex.

### C — Creative/research extension

- Mirror descent and Bregman divergences.
- Proximal methods, ADMM, coordinate descent.
- Convex-concave saddle-point problems and variational inequalities.
- Optimal transport and Wasserstein geometry.

### Read this, not everything

- **MML:** Chapter 7.
- **BV:** Chapters 2–5; then Chapter 9 on unconstrained minimization. The official site provides the book and examples, including Chapters 9–11. [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- **Boyd & Vandenberghe video lectures:** use only after working problems; videos are not a substitute for deriving KKT conditions.

### Derivation and proof targets

1. Prove that a positive weighted sum of convex functions is convex.
2. Derive gradient descent for a quadratic objective and relate convergence to the eigenvalues of the Hessian.
3. Form the Lagrangian for a constrained least-squares problem.
4. Derive KKT conditions for a simple inequality-constrained problem.
5. Explain why local minima are globally optimal in a convex problem, then state why this statement does not transfer to deep networks.

### Computation laboratory

- Implement gradient descent, Newton’s method, and projected gradient descent on small 2D objectives.
- Use CVXPY to solve a ridge, lasso, or constrained regression problem; verify KKT residuals numerically.
- Visualize level sets of a well-conditioned and ill-conditioned quadratic.

### Stop rule

You are ready when you can read an optimization objective and state: variables, objective, constraints, whether it is convex, what the relevant geometry is, and which guarantees you are allowed to claim.

### When to return here later

Return for **loss design**, **constrained RL**, **KL regularization**, **dual formulations**, **robust optimization**, **optimal transport**, and **algorithmic guarantees**.

---

# Month 8 — Stochastic and deep optimization

> **Question:** Why does SGD work at all in huge nonconvex networks, and why do schedules, noise, clipping, momentum, and adaptive scaling change outcomes?

### A — Anchor concepts

- Empirical risk and stochastic gradient estimators.
- Unbiasedness, variance, minibatches, gradient noise.
- SGD with momentum; exponential moving averages.
- Adam, RMSProp, AdamW, bias correction, decoupled weight decay.
- Learning-rate schedules: warmup, decay, cosine schedules.
- Gradient clipping, exploding/vanishing gradients.
- Curvature, Hessian intuition, saddle points, flat/sharp caveats.

### B — Bridge concepts to AI

- Foundation-model training uses stochastic estimates because full-dataset gradients are impossible at each step.
- Mixed precision changes the numerics of accumulation and scaling, not merely memory use.
- Gradient clipping and loss scaling are interventions on the update dynamics.
- “We trained longer” is not a meaningful scientific claim without data order, batch size, optimizer state, schedule, precision, and checkpoint-resume details.

### C — Creative/research extension

- Stochastic differential equation (SDE) approximations to SGD.
- Natural gradient, Fisher information, K-FAC, Shampoo.
- Sharpness, mode connectivity, loss landscapes.
- Scaling laws, compute-optimal training, and data-quality scaling.

### Read this, not everything

- **DLB:** Chapter 8, all sections on optimization for deep learning.
- **MML:** revisit Chapter 7 for the clean deterministic case.
- **Bottou, Curtis, Nocedal, “Optimization Methods for Large-Scale Machine Learning”** for the bridge from textbook optimization to stochastic practice.
- **Adam paper:** read the algorithm and bias-correction derivation; then compare Adam to AdamW rather than assuming the names differ only cosmetically.

### Derivation and proof targets

1. Show that a minibatch gradient is an unbiased estimator of the full empirical gradient under uniform sampling.
2. Derive momentum as an exponential moving average of gradients.
3. Write the Adam update, including first/second moment bias corrections, and explain why the early-step correction exists.
4. Explain how decoupled weight decay differs from simply adding \(\lambda\|\theta\|_2^2\) to the loss under adaptive optimization.
5. Use a Taylor approximation to explain why curvature affects stable learning rates.

### Computation laboratory

- Train the same small network with SGD, momentum, Adam, and AdamW; plot training/validation loss and gradient norms.
- Change only batch size while preserving or not preserving learning-rate scaling; observe update noise.
- Create a toy exploding-gradient RNN and rescue it with clipping.
- Instrument a training loop to log effective learning rates, gradient norms, weight norms, and loss-scale events.

### Stop rule

You are ready when you can look at a failed training curve and construct hypotheses that distinguish data bugs, scale/precision issues, optimizer instability, schedule mismatch, overfitting, and a genuine architecture limitation.

### When to return here later

Return for **pretraining recipes**, **fine-tuning instability**, **mixed precision**, **distributed optimizer sharding**, **scaling laws**, and **training reproducibility**.

---

# Month 9 — Numerical linear algebra and floating-point reality

> **Question:** Why can an algorithm be algebraically correct and still fail, slow down, overflow, underflow, or produce a different model on hardware?

### A — Anchor concepts

- Floating-point representation: sign, exponent, significand/mantissa; rounding.
- Overflow, underflow, cancellation, subnormals at a conceptual level.
- Machine epsilon, absolute/relative error.
- Forward error, backward error, stability.
- Conditioning versus stability.
- Mixed precision; accumulation precision; loss scaling.
- Quantization error, scales, zero-points, symmetric/asymmetric quantization.

### B — Bridge concepts to AI

- BF16 preserves FP32-like exponent range but sacrifices mantissa precision; FP16 has different trade-offs. This affects overflow/underflow and kernel strategy.
- Softmax requires numerical stabilization: 
  \[
  \operatorname{softmax}(z)_i=\frac{e^{z_i-\max_j z_j}}{\sum_k e^{z_k-\max_j z_j}}.
  \]
- Quantization is an approximation problem plus a systems problem. Lower bits matter only when kernel, memory movement, and error behavior align.
- Inference is often memory-bandwidth-bound, so representation precision changes not only model quality but physical data movement.

### C — Creative/research extension

- Backward-error analysis for matrix factorizations.
- Krylov subspace methods, preconditioners, iterative solvers.
- Stochastic rounding and low-precision optimizer states.
- Quantization-aware training, error feedback, and low-rank plus quantization co-design.

### Read this, not everything

- **DLB:** Chapter 4.
- **H:** Chapter 1, then selected chapters as needed. Higham’s book is specifically about finite-precision effects in numerical algorithms. [Higham overview](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/)
- **TB:** Lectures 1–5, then selected least-squares/eigenvalue lectures by title.
- **NVIDIA Transformer Engine documentation:** use later for actual FP8 implementation details, not as your first introduction to floating point.

### Derivation and proof targets

1. Derive stabilized softmax and show why subtracting a constant leaves probabilities unchanged.
2. Show a catastrophic-cancellation example and compare two algebraically equivalent formulas.
3. Define a condition number for a simple linear problem and demonstrate sensitivity numerically.
4. Derive uniform affine quantization/dequantization and its maximum rounding error under stated assumptions.
5. Explain the difference between error caused by the **problem** being ill-conditioned and error caused by the **algorithm** being unstable.

### Computation laboratory

- Run the same matrix multiplication and softmax computation in FP32, BF16/FP16 where available, and simulated INT8; measure error distributions.
- Implement naive and stabilized softmax; find an input where the naive version overflows.
- Quantize a small linear layer per-tensor and per-channel; compare output error and memory use.
- Measure arithmetic intensity for a simple matmul-like workload and connect it to a roofline estimate.

### Stop rule

You are ready when you can read “this method is numerically stable in BF16” and ask: stable for which operation, under what dynamic range, with what accumulator, scale policy, error metric, and hardware kernel?

### When to return here later

Return for **FlashAttention**, **FP8/FP4 training**, **quantization**, **kernel fusion**, **softmax**, **RMSNorm**, **distributed reductions**, and **serving performance claims**.

---

# Month 10 — Markov processes, dynamic programming, and sequential decision theory

> **Question:** What changes when a decision changes the future data you will see?

### A — Anchor concepts

- Markov chains, transition matrices, stationary distributions, mixing intuition.
- Markov decision processes (MDPs): states, actions, rewards, transitions, discount factor.
- Policies, trajectories, return, value function \(V^\pi\), action-value function \(Q^\pi\).
- Bellman expectation and Bellman optimality equations.
- Dynamic programming: policy evaluation, policy improvement, value iteration.
- Monte Carlo estimation and temporal-difference learning.
- On-policy versus off-policy distinction.

### B — Bridge concepts to AI

- In LLM RL, a “state” can be a prompt plus generated prefix, an action can be the next token, and an episode can be a response or multi-turn tool trajectory.
- The reward may arrive only at the final answer. Credit assignment asks how that terminal signal should influence earlier tokens/actions.
- The value function is a conditional expectation of future return, not a generic “score of goodness.”
- An agent environment is a transition system. Tool behavior, latency, hidden state, resets, and rewards are all mathematical design choices, not implementation scenery.

### C — Creative/research extension

- Partially observable MDPs (POMDPs), belief states, recurrent policies.
- Occupancy measures, linear-programming formulations of MDPs.
- Distributional RL, risk-sensitive RL, entropy-regularized RL.
- Off-policy evaluation and causal/doubly robust estimators.

### Read this, not everything

- **SB:** Chapters 2–6 in order: bandits, finite MDPs, dynamic programming, Monte Carlo methods, temporal-difference learning.
- **Puterman, *Markov Decision Processes*:** use selectively for a more formal theory reference.
- **B&H:** revisit conditional expectation and Markov-chain-related material when needed.

### Derivation and proof targets

1. Derive the Bellman expectation equation: 
   \[
   V^\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s].
   \]
2. Show why the Bellman optimality operator is a \(\gamma\)-contraction under the max norm for finite discounted MDPs.
3. Derive the Monte Carlo return and one-step TD error.
4. Construct a tiny MDP where an off-policy update differs from an on-policy update.
5. Map a single-turn language response to an episodic MDP and state what is missing from that abstraction.

### Computation laboratory

- Implement multi-armed bandits, tabular value iteration, Monte Carlo control, and Q-learning on a gridworld.
- Plot convergence and show how changing \(\gamma\) changes the effective planning horizon.
- Build a deterministic text-generation toy environment with a final exact-match reward.

### Stop rule

You are ready when you can turn a vague claim such as “we train an agent with reward” into a full MDP specification: state, action, transition, reward, horizon, discount, policy class, data source, and evaluation protocol.

### When to return here later

Return for **RLHF**, **agent training**, **tool-use environments**, **process rewards**, **credit assignment**, **curriculum learning**, and **offline/online RL debates**.

---

# Month 11 — Policy gradients, KL regularization, and the mathematics behind LLM post-training

> **Question:** How does a scalar reward alter a language model’s token distribution, and what mathematical compromises make that update stable enough to run at scale?

### A — Anchor concepts

- Log-derivative trick / score-function estimator:
  \[
  \nabla_\theta\mathbb{E}_{x\sim p_\theta}[R(x)]
  =\mathbb{E}_{x\sim p_\theta}[R(x)\nabla_\theta\log p_\theta(x)].
  \]
- REINFORCE, baselines, variance reduction, advantage.
- Actor–critic intuition; generalized advantage estimation (GAE) at a working level.
- Importance sampling and policy mismatch.
- KL-regularized RL objective.
- Trust regions, PPO clipped surrogate objective.
- Preference objectives and the mathematical relationship between reward models, pairwise comparisons, and direct preference optimization at a conceptual/derivational level.

### B — Bridge concepts to AI

- For an LM, the sequence log-probability decomposes over tokens:
  \[
  \log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).
  \]
  This is how a response-level reward sends gradient to individual token decisions.
- KL regularization prevents a reward model or verifier from pulling the policy too far from a reference model’s behavior.
- PPO stabilizes updates by restricting the effective change in action probabilities; it is not a magical reward-hacking cure.
- DPO-style methods move part of the reward-model-plus-RL pipeline into an offline preference objective. They trade online exploration for training simplicity and data dependence.
- GRPO/RLVR-style methods exploit group rollouts and verifiable rewards, but their statistical and exploration behavior still depends on the generator, reward, environment, and sampling policy.

### C — Creative/research extension

- Maximum-entropy RL and control-as-inference.
- Natural policy gradient and Fisher geometry.
- Mirror descent policy optimization.
- Offline RL, conservative objectives, behavior regularization.
- Preference learning identifiability, reward-model misspecification, and Goodhart effects.

### Read this, not everything

- **SB:** Chapter 13, *Policy Gradient Methods*, after completing Chapters 2–6; then selected Chapters 9–10 for function approximation/off-policy background.
- **Williams (1992):** REINFORCE.
- **TRPO:** for constrained policy updates.
- **PPO:** derive the clipped surrogate rather than memorizing the min expression.
- **InstructGPT:** for the classical LLM RLHF pipeline.
- **DPO:** for a modern offline preference objective.
- **Your guide’s Part VIII and VIII-A:** for system implementation, rollout engines, reward services, and current research disputes.

### Derivation and proof targets

1. Derive REINFORCE using the log-derivative trick.
2. Prove that subtracting an action-independent baseline does not bias the policy-gradient estimator.
3. Derive a sequence-level LM policy-gradient objective using autoregressive log probabilities.
4. Write the KL-regularized objective and explain the role of \(\beta\).
5. Analyze PPO’s probability ratio \(r_t(\theta)\) and explain what clipping suppresses and what it does **not** guarantee.
6. Derive the qualitative DPO connection from a Bradley–Terry preference model and a KL-regularized optimal policy; use the original paper for the exact assumptions.

### Computation laboratory

- Implement REINFORCE with and without a baseline on a small discrete environment; quantify variance.
- Implement a KL-regularized bandit policy update and vary \(\beta\).
- Build a toy sequence generator with a deterministic verifier, sample groups of candidates, and compare raw-return versus normalized/group-relative updates.
- Implement DPO on synthetic pairwise preferences and compare it to supervised preference imitation.

### Stop rule

You are ready when you can inspect an LLM-RL paper and identify: the policy distribution, trajectory, reward source, credit-assignment estimator, reference policy, KL mechanism, on/off-policy status, rollout version, and the specific way reward hacking could occur.

### When to return here later

Return for **RLHF**, **RLAIF**, **PPO**, **DPO**, **GRPO**, **RLVR**, **reasoning training**, **agent RL**, **reward models**, and **verifier design**.

---

# Month 12 — Dynamical systems, generalization, and research-math synthesis

> **Question:** Which mathematical language do you need when a paper stops being “just a Transformer paper” and starts involving state evolution, stability, scaling, or a new theoretical claim?

This month has a **mandatory synthesis core** and a choice of **two research extensions**. Choose extensions based on the research direction you want to pursue rather than trying to consume all of them in four weeks.

### A — Mandatory synthesis core

- Ordinary differential equations (ODEs): state, derivative, initial value problem.
- Linear time-invariant systems:
  \[
  \dot h(t)=Ah(t)+Bx(t), \qquad y(t)=Ch(t).
  \]
- Matrix exponential and discretization intuition.
- Stability through eigenvalues/real parts; why discretization matters.
- Convolutional view of linear state-space systems.
- Generalization vocabulary: empirical risk, population risk, concentration intuition, capacity/complexity, and the limits of simplistic theory claims.
- Research methodology: specify assumptions, define a claim, choose a counterexample, make an ablation that tests mechanism instead of merely reporting a number.

### B — Bridge concepts to AI

- Mamba/state-space models use a discretized state evolution rather than storing all past keys/values explicitly. You need linear systems, matrix exponentials, discretization, and stability language to read the core papers rather than treating them as a black box.
- Inference systems are dynamic systems too: queues, batching policies, cache state, and control loops evolve over time.
- Scaling-law and generalization claims are mathematical-statistical statements that must identify the data regime, model family, budget, and held-out distribution.

### C — Choose two research extensions

#### Extension 1: State-space models and control

Study:
- zero-order hold discretization;
- controllability and observability;
- transfer functions and convolution kernels;
- selective/state-dependent parameterization at a conceptual level.

Use for: **Mamba, Mamba-2, Jamba, sequence-model alternatives, control agents, robotics.**

References: a linear-systems text such as Chen, *Linear System Theory and Design*, plus the Mamba and Mamba-2 papers.

#### Extension 2: Statistical learning theory and generalization

Study:
- Hoeffding/Chernoff-style concentration;
- uniform convergence intuition;
- VC dimension and Rademacher complexity;
- PAC-Bayes as a research-reading vocabulary.

Use for: **scaling laws, benchmark claims, generalization theory, compression bounds, uncertainty.**

References: Shalev-Shwartz and Ben-David, *Understanding Machine Learning*, selected chapters; Mohri, Rostamizadeh, Talwalkar, *Foundations of Machine Learning*, selected chapters.

#### Extension 3: Measure-theoretic probability and stochastic processes

Study:
- sigma-algebras, measurable functions, Lebesgue expectation;
- convergence almost surely/in probability/in distribution;
- martingales and filtrations at an introductory level.

Use for: **advanced probability papers, diffusion models, stochastic optimization theory, rigorous RL.**

References: Durrett, *Probability: Theory and Examples*, selected early chapters; do this only after Months 4–5 feel comfortable.

#### Extension 4: Information geometry, natural gradient, and optimal transport

Study:
- Fisher information matrix;
- Riemannian metric intuition;
- natural gradient;
- Wasserstein distance and transport plans.

Use for: **second-order optimization, policy optimization, distribution shift, generative modeling.**

References: Amari’s work for information geometry; Peyré and Cuturi, *Computational Optimal Transport*, selected introduction and algorithm chapters.

#### Extension 5: Random matrix theory and high-dimensional statistics

Study:
- spectral distributions, concentration of random matrices;
- Marchenko–Pastur intuition;
- high-dimensional covariance estimation.

Use for: **initialization, training dynamics, scaling, large-model theory.**

References: Vershynin, *High-Dimensional Probability*, selected chapters; specialist papers only after the linear algebra/probability core.

### Derivation and proof targets

1. Solve a scalar linear ODE and write its discretized recurrence.
2. Derive the matrix-exponential solution for a simple LTI system.
3. Show how a linear recurrence creates a convolution over past inputs.
4. Write a one-page argument distinguishing empirical performance from a generalization claim.
5. For one paper from this guide, list every mathematical assumption it needs but does not foreground.

### Computation laboratory

- Implement a stable and unstable scalar recurrence; visualize the effect of eigenvalue magnitude.
- Discretize a small linear state-space system and compare recursive versus convolutional computation.
- Reproduce a tiny SSM-like sequence task and compare memory/computation with causal attention on the same controlled setup.
- Write a mini research proposal: one claim, one hypothesis, one falsifier, one metric, one synthetic diagnostic, one real-world evaluation.

### Stop rule

You are ready to move from “math curriculum” to sustained research when you can meet an unfamiliar paper, classify its mathematical dependencies, and build a two-week catch-up plan rather than feeling that the paper demands a second undergraduate degree.

---

## I-A.4 The AI-mathematics dependency map

Use this table when a topic in the main guide feels opaque. It tells you which mathematics to revisit rather than telling you to “study more math.”

| AI topic | Anchor mathematics | Bridge derivation you should be able to do | Research extension that pays off |
|---|---|---|---|
| Token embeddings and MLPs | Linear maps, bases, norms | Shapes for \(XW+b\); gradient of affine layer | Representation geometry, random matrices |
| Self-attention / GQA / MLA | Matrix multiplication, dot products, softmax calculus | \(QK^T/\sqrt{d}\), softmax cross-entropy, KV-cache size | Kernel methods, low-rank approximation, information theory |
| MoE routing | Categorical distributions, softmax, constrained optimization | Router probabilities, load-balancing loss, expected compute | Optimal transport, stochastic routing, combinatorial optimization |
| Mamba / SSMs | ODEs, linear systems, eigenvalues, convolution | discretized recurrence and stability | Control theory, operator theory, signal processing |
| Pretraining | Probability, MLE, information theory, SGD | NLL/cross-entropy and \(p-y\) gradient | scaling laws, statistical learning theory |
| Quantization / FP8 | Floating point, error analysis, conditioning | affine quantization error, stable softmax | stochastic rounding, backward error analysis |
| Distributed training | Linear algebra, numerical reduction error, optimization | all-reduce mean gradient, sharded tensor shapes | communication lower bounds, asynchronous optimization |
| Inference serving | Arithmetic intensity, queues, probability | KV-cache memory estimate, roofline bound | queueing theory, online scheduling, control |
| RLHF / DPO / PPO | MDPs, expectation, KL, constrained optimization | REINFORCE, baseline, PPO ratio, DPO assumptions | control-as-inference, offline RL, information geometry |
| Agent RL | POMDPs, causal/sequential statistics | state/action/transition specification, credit assignment | causal inference, off-policy evaluation, process supervision theory |

---

## I-A.5 The derivation portfolio: twelve things you should own

By the end of the program, maintain clean handwritten or typeset derivations of the following. These are not rote rituals. They are a personal emergency toolkit for reading papers.

1. Matrix multiplication as composition of linear maps, with shapes.
2. Orthogonal projection and least squares normal equations.
3. SVD geometry and low-rank approximation intuition.
4. Backpropagation through an affine layer and nonlinear activation.
5. Softmax + cross-entropy gradient \(p-y\).
6. Autoregressive likelihood factorization and sequence NLL.
7. Cross-entropy / KL decomposition and non-negativity of KL.
8. Gradient descent on a quadratic, with eigenvalue-dependent convergence.
9. SGD unbiasedness and minibatch variance intuition.
10. Stabilized softmax and a floating-point failure example.
11. Bellman expectation equation and contraction intuition.
12. REINFORCE with an unbiased baseline and sequence-level policy gradient.

### Three “research-extra” derivations

Add these if your research focus calls for them:

13. PPO clipped objective and its local policy-ratio interpretation.
14. LTI state-space discretization and convolutional equivalence.
15. A LoRA rank-\(r\) update’s parameter/memory cost versus full fine-tuning.

---

## I-A.6 What you can safely postpone

These topics are valuable, but they are **not prerequisites** for beginning serious AI engineering research unless your chosen question directly invokes them.

- Abstract algebra, category theory, and topology.
- Full measure theory before you have strong ordinary probability intuition.
- Differential geometry beyond gradients/Jacobians/Fisher intuition.
- Functional analysis beyond basic normed-space language.
- Advanced PDE theory.
- Algebraic geometry.
- Deep stochastic-calculus machinery.

Do not misunderstand this as dismissal. These subjects become essential in some research areas. The rule is simply: **pull mathematics in response to a concrete research dependency.**

```text
Paper introduces diffusion SDEs             → probability + stochastic processes + SDE basics
Paper introduces SSM stability theorem      → linear systems + spectral theory
Paper introduces optimal-transport loss     → convex duality + transport
Paper introduces PAC-Bayes bound            → probability + KL + concentration
Paper introduces natural-gradient method    → Fisher information + differential geometry
Paper introduces quantization guarantee     → norms + error bounds + numerical linear algebra
```

That is how a researcher uses books: not as a ceremonial mountain climb, but as a map room.

---

## I-A.7 Research-paper reading protocol for mathematics

When reading a technical paper, make a **math dependency card** before reading the experiments.

```text
Claim:
What exactly is improved or proved?

Objects:
What are the vectors, matrices, distributions, states, parameters, and random variables?

Objective:
Write the loss, constraint, or recurrence in your notation.

Assumptions:
Which differentiability, independence, convexity, stationarity, boundedness, or precision assumptions are required?

Core lemma / mechanism:
Which equation is doing the real work?

Limit case:
What happens if a coefficient becomes zero, a temperature goes to infinity, rank becomes full, or the system becomes deterministic?

Counterexample:
Can you construct a tiny case where the claim would fail?

Book return path:
Which textbook chapter supplies the missing fact?
```

This protocol is deliberately slower than reading a blog post. It is also how you avoid being impressed by notation that has not yet said anything.

---

## I-A.8 Recommended assessment gates

At the end of Months 3, 6, 9, and 12, take a two-day **closed-notes synthesis exam** that you write for yourself.

### Gate 1 — after Month 3: differentiable computation

- Derive gradients for a two-layer MLP.
- Explain attention tensor shapes.
- Implement and finite-difference-check a tiny autograd engine.
- Diagnose three deliberately broken gradient calculations.

### Gate 2 — after Month 6: probabilistic modeling

- Derive language-model NLL and cross-entropy.
- Explain entropy, KL, and perplexity with a discrete numerical example.
- Design a valid train/validation/test protocol for a small experiment.
- Critique a fake benchmark claim with missing uncertainty information.

### Gate 3 — after Month 9: optimization under numerical constraints

- Analyze a convex quadratic’s conditioning and gradient-descent behavior.
- Explain AdamW and gradient clipping precisely.
- Demonstrate a softmax overflow and repair it.
- Quantize a layer and report error, memory, and latency implications separately.

### Gate 4 — after Month 12: sequential learning and research reasoning

- Specify an MDP for a tool-using language agent.
- Derive REINFORCE with a baseline and write the sequence policy gradient.
- Explain KL regularization’s role in PPO/RLHF.
- Read one SSM, RL, or systems paper and produce a complete math dependency card plus a falsification experiment.

---

## I-A.9 Minimum references, live links, and chapter return map

### Main texts

1. [Mathematics for Machine Learning](https://mml-book.github.io/) — use Chapters 2–7 as your first-pass mathematical core.
2. [Deep Learning, Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/) — Chapters 2–4 and 8 form the deep-learning mathematics return shelf.
3. [Understanding Deep Learning, Simon J. D. Prince](https://udlbook.github.io/udlbook/) — use its mathematics material and Appendices B/C when an ML derivation needs more detail.
4. [Introduction to Probability, Blitzstein and Hwang](https://projects.iq.harvard.edu/stat110/home) — Chapters 1–7 for probability discipline.
5. [Probabilistic Machine Learning: An Introduction, Kevin Murphy](https://probml.github.io/pml-book/book1.html) — second pass for probabilistic models and inference.
6. [Elements of Information Theory, Cover and Thomas](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — Chapter 2 first, then follow research needs.
7. [Convex Optimization, Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — Chapters 2–5 and 9 core; Chapters 10–11 later.
8. [Numerical Linear Algebra, Trefethen and Bau](https://people.maths.ox.ac.uk/trefethen/text.html) — Lectures 1–5 core, then targeted return by method.
9. [Accuracy and Stability of Numerical Algorithms, Nicholas Higham](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/) — begin with Chapter 1.
10. [Reinforcement Learning: An Introduction, Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) — Chapters 2–6 and 13 core.

### Useful problem and implementation companions

- [3Blue1Brown, Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) for visual orientation, never as the only source.
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for identity lookup after you have derived an identity at least once.
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) for implementation behavior.
- [micrograd](https://github.com/karpathy/micrograd) for minimal reverse-mode autodiff.
- [CVXPY](https://www.cvxpy.org/) for constrained-optimization experiments.

---

## I-A.10 Final note: mathematics is a navigation instrument, not a collection hobby

The goal is not to become the person who can list every theorem adjacent to AI. The goal is to become the person who can encounter a new model, optimization method, accelerator claim, or RL objective and calmly ask:

1. **What are the objects?**
2. **What is the objective or state evolution?**
3. **What assumptions make the derivation legal?**
4. **What approximation enters during implementation?**
5. **Which textbook chapter repairs the exact gap in my understanding?**

Once those questions become instinctive, mathematics stops feeling like a prerequisite dungeon. It becomes the x-ray machine you carry into every new AI paper.

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

## 4.4 Scaling laws and compute-optimal training

Scaling laws model how loss changes with model parameters, data, and compute. The Chinchilla result argued that many earlier large language models were undertrained relative to their parameter count and that, under its studied regime, model size and training-token count should scale together. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

The enduring lesson is not “use exactly one token-to-parameter ratio forever.” It is:

> A training budget must be allocated jointly across model size, data volume, architecture, sequence length, and expected inference demand.

If a model will receive massive inference traffic, a smaller model trained longer may be economically attractive because inference costs recur for every request. [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)

## 4.5 Memory accounting during training

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

## 4.6 Distributed training parallelisms

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

## 4.7 Communication is part of the model

Collectives matter:

- **all-reduce:** aggregate gradients and return result to all participants;
- **all-gather:** collect shards so each participant obtains the full logical tensor;
- **reduce-scatter:** reduce then distribute shards;
- **all-to-all:** every device exchanges different slices with every other device, central to MoE routing.

A model may be mathematically elegant but operationally poor if communication dominates compute.

## 4.8 Mixed precision training

Mixed precision uses lower-precision formats for appropriate operations while preserving enough high-precision state for stability.

Typical concepts:

- BF16 / FP16 activations and matrix multiplies;
- FP32 accumulations where necessary;
- loss scaling for FP16 regimes;
- FP8 recipes with scaling metadata and carefully designed kernels;
- numerical monitoring for overflow, NaNs, and loss spikes.

Modern accelerators increasingly support FP8 and FP4 pathways, but the recipe is hardware- and model-dependent. Do not reduce “mixed precision” to changing a dtype flag.

## 4.9 Checkpointing and fault tolerance

At scale, training is an operations problem:

- save model, optimizer, scheduler, tokenizer, and data-progress state;
- support resuming after preemption or machine failure;
- shard checkpoints to avoid one process becoming an I/O bottleneck;
- validate that resumed runs reproduce expected behavior;
- manage versioning of data and code.

A model checkpoint without its tokenizer, training configuration, data provenance, and evaluation context is an incomplete artifact.

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

## 6.13 Inference research you should watch

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

## 7.8 Evaluation protocol for quantized models

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

> **What this part is for:** Post-training is often described as a tidy ladder, “SFT → RLHF → reasoning RL.” That picture is too shallow. This chapter reconstructs the lineage from classical reinforcement learning to modern LLM post-training; derives the central objectives; explains what actually runs on a cluster; separates offline preference optimization, online RLHF, RL with verifiable rewards, and multi-turn agent RL; and then maps the live research questions. Read it slowly. This is a field, not a feature checkbox.

## 8.1 The central question: what behavior are we trying to optimize?

Pretraining optimizes a predictive objective over text and other data:

\[
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}}\left[\sum_t \log \pi_\theta(x_t\mid x_{<t})\right].
\]

That teaches a distribution to continue sequences. It does **not** directly tell the model what a particular user means by helpful, how to trade off correctness against brevity, whether a tool action succeeds, whether code passes tests, or whether an answer follows a policy.

Post-training introduces a different question:

\[
\text{Which completions, trajectories, or environment outcomes should become more probable?}
\]

The answer depends on the available supervision:

| Supervision available | Typical method family | What is optimized |
|---|---|---|
| Demonstrations | SFT / behavior cloning | likelihood of a target response |
| Pairwise human or AI preferences | reward modeling + RLHF, DPO-family | relative desirability |
| Scalar evaluator or learned judge | online RLHF / RLAIF | expected evaluator reward |
| Deterministic checker, test, proof, simulator | RLVR / GRPO / PPO / search | verified task outcome |
| Multi-step tool environment | agent RL | terminal outcome plus environment feedback |
| No reliable external feedback | self-training, self-feedback, process models, uncertainty methods | a carefully audited proxy |

**Key distinction:** “post-training” is the umbrella. **RLHF**, **DPO**, **RLAIF**, **RLVR**, and **agent RL** are not interchangeable names for the same operation.

---

## 8.2 Historical lineage: how the field got here

### Before LLMs: the mathematical spine

| Era | Development | Why it still matters for LLMs |
|---|---|---|
| 1950s | Bellman’s dynamic programming and optimal-control view | value functions and recursive returns |
| 1980s–1990s | temporal-difference learning, Q-learning, REINFORCE | bootstrapping, policy gradients, exploration, variance reduction |
| 2010s | deep RL, actor–critic methods, TRPO/PPO | neural policies, stable trust-region-like updates, distributed rollout systems |
| 2017 | RL from human preferences | learn a reward signal from comparisons rather than hand-writing it |
| 2019–2020 | language-model preference fine-tuning and summarization from human feedback | showed reward modeling plus policy optimization worked for text generation |
| 2022 | InstructGPT, helpful-harmless RLHF, Constitutional AI | made post-training a central LLM pipeline stage and introduced large-scale AI feedback |
| 2023 | DPO and process supervision | simplified offline preference learning; made reasoning-step supervision a first-class object |
| 2024 | GRPO / DeepSeekMath, open RLHF systems | group-relative verifier-based optimization and scalable open infrastructure |
| 2025 | DeepSeek-R1, DAPO, agent RLVR, reward-model-as-reasoner | reasoning RL and environment-based training became major open research directions |
| 2026 | rollout strategy, replay/prioritization, off-policy/asynchronous training, agent and multimodal environments | the focus shifts from “which optimizer?” toward data, environments, verifiers, system throughput, and credible evaluation |

### The 2024 → 2026 shift in plain language

**Around 2024**, the open LLM post-training conversation centered heavily on instruction tuning and **offline preference optimization**. DPO and its relatives were attractive because they avoided the operational complexity of online PPO-style RL. GRPO appeared in open reasoning research through DeepSeekMath, but verifier-based reasoning RL was still a rapidly developing recipe rather than settled infrastructure.

**By 2025–2026**, the center of gravity in open research moved toward **online rollout-based training** when an evaluator can verify outcomes: mathematics, code, formal reasoning, tool use, and increasingly multi-step agent tasks. The hard part became less “Can we write a policy-gradient loss?” and more “Can we generate enough informative trajectories, score them honestly, keep rollouts sufficiently on-policy, schedule inference and training efficiently, prevent reward exploitation, and prove the gains transfer?” DeepSeek-R1 popularized this shift in the open literature, while DAPO, Agent-RLVR, asynchronous-RLHF work, and modern RL platforms made the systems questions explicit. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [DAPO](https://arxiv.org/abs/2503.14476) · [Agent-RLVR](https://arxiv.org/abs/2506.11425) · [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)

---

## 8.3 Classical RL from first principles

### 8.3.1 Markov decision processes

A standard episodic Markov decision process (MDP) is:

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0).
\]

- \(\mathcal{S}\): states
- \(\mathcal{A}\): actions
- \(P(s' \mid s,a)\): transition dynamics
- \(r(s,a,s')\): reward
- \(\gamma \in [0,1]\): discount factor
- \(\rho_0\): initial-state distribution

A policy \(\pi_\theta(a\mid s)\) maps a state to a distribution over actions. A trajectory is:

\[
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T).
\]

The return from time \(t\) is:

\[
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
\]

The objective is expected return:

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0].
\]

### 8.3.2 Why language-model RL is not a clean MDP

For a language model, we often write:

```text
state  = prompt + tokens generated so far + tool observations + environment state
action = next token, a tool call, or a structured action
policy = the next-token distribution of the LM
reward = human preference, judge score, verifier result, test result, or environment outcome
```

But an LLM setting is often **partially observable**:

- the model does not see the user’s true intent;
- tool state may be hidden or summarized;
- the environment can change outside the context window;
- truncation and compaction mean the textual “state” may omit history.

So the more honest formal model is frequently a POMDP. The model conditions on an observation or belief state, not necessarily the full world state. This matters especially for agents: a browser screenshot, a file tree, and a compacted chat history are all lossy observations of the task state.

### 8.3.3 Value functions and Bellman recursion

The state-value function is:

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s],
\]

and the action-value function is:

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a].
\]

Their difference defines the advantage:

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

The Bellman expectation equation is:

\[
V^\pi(s)=\mathbb{E}_{a\sim\pi,\,s'\sim P}[r(s,a,s')+\gamma V^\pi(s')].
\]

This is the deep idea behind critics: instead of waiting for a full trajectory return, estimate how promising the current state is. In a text-generation task with one terminal reward, a learned critic attempts to estimate whether a partial completion is likely to end well.

### 8.3.4 REINFORCE: the seed of policy gradients

For a stochastic policy, the likelihood-ratio trick gives:

\[
\nabla_\theta J(\theta)=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
G_0\nabla_\theta\log p_\theta(\tau)
\right].
\]

Because a trajectory probability factors over actions:

\[
\log p_\theta(\tau)=\sum_t\log\pi_\theta(a_t\mid s_t)+\text{terms independent of }\theta,
\]

we get:

\[
\nabla_\theta J(\theta)=
\mathbb{E}\left[
\sum_t G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
\]

For language models, each token is an action. If a whole completion receives reward \(R\), a basic sequence-level estimator is:

\[
\nabla_\theta J(\theta)\approx
R\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
\]

This tells the model: *increase probability of every action in a high-reward sampled trajectory; decrease it for a low-reward trajectory.*

### 8.3.5 Why the naïve estimator is painful

REINFORCE is unbiased under its assumptions, but it has high variance. A trajectory can succeed due to one crucial action while every previous token receives equal credit. This is the **credit-assignment problem**.

A baseline \(b(s)\) reduces variance without changing the expected gradient:

\[
\mathbb{E}\left[(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
\]

Choosing \(b(s)=V^\pi(s)\) yields the advantage estimator. PPO, GRPO, RLOO, critic-free group methods, and many modern variants can be understood as different practical ways to obtain a usable learning signal while preventing destructive updates.

### 8.3.6 On-policy, off-policy, and why this matters operationally

- **On-policy:** rollouts come from the same or very recent policy being optimized. The gradient signal is most directly tied to current behavior, but generation is expensive and synchronization slows the cluster.
- **Off-policy:** train on data from an older policy, a behavior policy, or a replay buffer. This can improve utilization but requires correcting or tolerating policy mismatch.

The importance ratio is:

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
\]

When policy versions drift too far apart, this ratio becomes unstable and old trajectories become misleading. This is why LLM RL systems care about rollout-policy versions, staleness, asynchronous queues, and policy synchronization. The classic clean equation grows teeth in a distributed system.

**Canonical foundations:** [Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) · [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) · [Q-learning](https://link.springer.com/article/10.1007/BF00992698) · [PPO](https://arxiv.org/abs/1707.06347)

---

## 8.4 From language modeling to RL objectives

### 8.4.1 A completion is an action sequence

For a prompt \(x\) and completion \(y=(y_1,\ldots,y_T)\), an autoregressive policy is:

\[
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).
\]

The log probability is additive:

\[
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
\]

This is why a sequence-level reward can be turned into token-level gradients. The model samples a complete answer; a reward process scores it; the training loop raises or lowers probability of the sampled token sequence.

### 8.4.2 The KL-regularized objective

Post-training usually should not maximize reward without restraint. A common objective is:

\[
\max_\pi\;
\mathbb{E}_{x\sim\mathcal{D},y\sim\pi(\cdot\mid x)}
\left[r(x,y)\right]
-
\beta\,D_{KL}\left(\pi(\cdot\mid x)\;||\;\pi_{\text{ref}}(\cdot\mid x)\right).
\]

The reference policy is typically an SFT checkpoint or a frozen prior policy. The KL term is not decorative. It combats reward overoptimization, linguistic degradation, narrow-mode collapse, and unwanted drift from useful pretrained behavior.

There is no universally correct \(\beta\). A weak KL constraint can let the policy exploit reward-model bugs; an overly strong one makes training inert. Tracking reward without tracking KL, entropy, and held-out capability is not serious RL experimentation.

### 8.4.3 Reward, return, advantage, and loss are not synonyms

| Object | Meaning |
|---|---|
| reward | scalar feedback from an evaluator at one or more points |
| return | cumulative discounted future reward |
| value | expected return from a state |
| advantage | relative quality of an action compared with a baseline |
| policy loss | differentiable surrogate used to update parameters |
| reward model | learned predictor of preference, not the ground truth itself |
| verifier | an executable evaluator; may still be incomplete or exploitable |

Keeping these distinct will prevent most early confusion.

---

## 8.5 Before RL: SFT, instruction tuning, and behavior cloning

### 8.5.1 SFT is supervised behavior cloning

Given demonstrations \((x,y^*)\), supervised fine-tuning minimizes:

\[
\mathcal{L}_{\text{SFT}}(\theta)=
-\mathbb{E}_{(x,y^*)}\left[\log\pi_\theta(y^*\mid x)\right].
\]

SFT is stable, cheap relative to online RL, and usually supplies the warm start for a post-training stack. It changes the model’s default response distribution toward desired formats, instructions, tool syntax, and task behaviors.

### 8.5.2 Its limitation is distribution mismatch

SFT learns from **demonstrated** trajectories. At deployment, the model samples from its own policy and may visit prefixes absent from the demonstration dataset. This is exposure bias in a broad sense: mistakes alter the next state, and the training distribution did not necessarily include recovery from those mistakes.

RL can optimize behavior on the model’s own sampled outputs. That is valuable, but it introduces expensive rollouts and reward-design risks. The field’s history is largely a long negotiation between these two realities.

### 8.5.3 Instruction tuning was a major bridge

Instruction-tuning work showed that fine-tuning on diverse tasks phrased as instructions improves generalization to new instructions. [FLAN / Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) is an important bridge between pretraining and alignment-style post-training.

---

## 8.6 Classical RLHF: reward modeling plus online policy optimization

### 8.6.1 Where RLHF came from

The foundational human-preference setting asks annotators to choose which of two trajectory segments they prefer. This replaces a hand-specified reward with learned preferences. [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) established the general pattern before LLM alignment. Language-model applications followed with preference-based fine-tuning in 2019 and summarization from human feedback in 2020. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) · [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)

### 8.6.2 The canonical RLHF pipeline

```text
1. Start with a pretrained base model.
2. Instruction-tune it on demonstrations → policy π_ref / warm-start policy.
3. Sample multiple candidate responses to prompts.
4. Collect pairwise preferences: y⁺ preferred to y⁻.
5. Train a reward model rφ(x, y).
6. Generate on-policy rollouts from the current policy πθ.
7. Score rollouts with rφ, usually with a KL penalty to π_ref.
8. Update πθ with PPO or another online RL algorithm.
9. Evaluate quality, safety, calibration, regression, and reward exploitation.
10. Refresh prompts, labels, and/or the reward model when the policy distribution shifts.
```

InstructGPT made this pipeline central to public understanding of LLM alignment: human-written demonstrations establish an SFT policy; labelers rank model outputs; a reward model is trained; then PPO optimizes the policy using reward-model scores. [InstructGPT](https://arxiv.org/abs/2203.02155)

### 8.6.3 Pairwise reward-model training

For prompt \(x\), preferred completion \(y^+\), and rejected completion \(y^-\), a Bradley–Terry reward model assigns:

\[
P(y^+\succ y^-\mid x)=
\sigma\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right).
\]

The negative log-likelihood loss is:

\[
\mathcal{L}_{\text{RM}}(\phi)=
-\mathbb{E}\left[
\log \sigma\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right)
\right].
\]

The reward model learns a *ranking-compatible scalar proxy*. It does not discover human values as a physical law. It generalizes from finite, noisy, sometimes inconsistent labels. The policy will search for high-scoring outputs, including pathological regions the label dataset never covered.

### 8.6.4 PPO in LLM RLHF

PPO constrains updates relative to the rollout policy. Let

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
\]

The clipped surrogate objective is:

\[
\mathcal{L}^{\text{CLIP}}(\theta)=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)A_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

The clip prevents a sampled trajectory from causing an arbitrarily large policy-ratio update. In LLM RLHF, the actual loss commonly also contains a value-function term, an entropy term, and a KL penalty or KL-shaped reward.

A useful mental model:

```text
PPO = “Improve on sampled rollouts, but do not move so far from the policy that generated them that our local estimate becomes untrustworthy.”
```

### 8.6.5 What makes PPO operationally difficult for LLMs

Traditional RL textbooks show one environment, one policy, and one learner. LLM RLHF needs at least:

- a large rollout policy for generation;
- a trainable actor policy;
- usually a frozen reference policy;
- often a value model / critic;
- a reward model or an external judge;
- distributed prompt queues and policy-version management;
- checkpointing, data provenance, and evaluation.

A naïve arrangement can require keeping multiple huge models resident or repeatedly reshaping/sharding them between generation and training. This is why the systems literature around RLHF is real research, not plumbing. [HybridFlow](https://arxiv.org/abs/2409.19256) studies this distributed dataflow explicitly.

---

## 8.7 RLAIF, Constitutional AI, and scalable evaluators

Human labels are expensive and limited. A natural response is to use models as evaluators.

### 8.7.1 RLAIF

**Reinforcement Learning from AI Feedback** uses model-generated preference or critique signals instead of, or alongside, human labels. It lowers marginal labeling cost but shifts the question to evaluator quality: why should the model judge be trusted, calibrated, robust to prompt injection, and not merely amplify the policy’s existing blind spots?

### 8.7.2 Constitutional AI

Constitutional AI makes the source of AI feedback more explicit: a set of written principles guides critique and revision, followed by AI-preference modeling and RL. The original paper is important because it combines supervised self-critique/revision with RLAIF and frames the constitution as an inspectable source of normative guidance. [Constitutional AI](https://arxiv.org/abs/2212.08073)

### 8.7.3 Generative and reasoning reward models

Scalar reward heads are compact but opaque. A newer direction asks a judge model to reason about the response before producing a score or verdict. Such **generative reward models** may be more expressive, but they add their own inference cost, calibration problems, and susceptibility to persuasive but wrong rationales. [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)

**Research question:** When does a stronger judge improve the policy, and when does it simply create a more elaborate exploitable proxy?

---

## 8.8 Direct preference optimization and the offline preference family

### 8.8.1 Why DPO mattered

PPO-style RLHF is online and systems-heavy. DPO showed that, under a KL-regularized reward-model formulation, one can express the optimal-policy condition as a direct classification objective on preference pairs, avoiding a separate learned reward model and an online PPO loop for the core preference stage. [DPO](https://arxiv.org/abs/2305.18290)

For \((x,y_w,y_l)\), a representative DPO loss is:

\[
\mathcal{L}_{\text{DPO}}(\theta)=
-\mathbb{E}\left[
\log\sigma\left(
\beta
\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right)
\right].
\]

It increases the preferred response’s relative log-probability over the rejected response, compared with the reference policy.

### 8.8.2 What DPO does not eliminate

DPO simplifies a particular alignment objective. It does **not** remove:

- the need for high-quality preference data;
- reward misspecification in the labels;
- distribution shift beyond the offline pair dataset;
- evaluation challenges;
- the need for online exploration when the desired behavior requires discovering new successful trajectories.

DPO is often an excellent choice for preference alignment. It is not a universal substitute for online RL in verifiable or interactive environments.

### 8.8.3 A map of important offline variants

Do not try to memorize dozens of names. Group them by which dependency they remove or which feedback type they accept.

| Method | Main idea | Why study it |
|---|---|---|
| DPO | pairwise preference objective with a reference policy | canonical derivation and baseline |
| IPO | adjusts preference optimization to address overfitting/pathologies | understand objective design and margins |
| KTO | learns from binary desirable/undesirable feedback | useful when paired comparisons are unavailable | 
| ORPO | combines SFT-like learning and odds-ratio preference optimization without a reference model | illustrates monolithic, reference-free objectives |
| SimPO | uses length-normalized average log probability and no reference model | illustrates sequence-length and reference-cost design |
| RSO / online DPO | incorporate sampling or online data | bridge offline objectives and rollout-based learning |
| DRO / single-trajectory methods | learn from scalar feedback rather than pairs | relevant to real product feedback logs |

Canonical papers: [KTO](https://arxiv.org/abs/2402.01306) · [ORPO](https://arxiv.org/abs/2403.07691) · [SimPO](https://arxiv.org/abs/2405.14734) · [Offline Regularised RL / Direct Reward Optimisation](https://arxiv.org/abs/2405.19107)

### 8.8.4 Offline versus online is the deeper axis

The central design question is often not “PPO or DPO?” but:

```text
Can the training data already contain the successful behavior we want?
    Yes → offline preference / supervised methods may be sufficient and simpler.
    No  → online sampling, search, environment interaction, or curriculum may be necessary.
```

If you need the model to discover a novel proof, a passing software patch, or a multi-step browser trajectory, offline pairs may not contain enough successful evidence. Online rollout generation becomes attractive despite its cost.

---

## 8.9 Outcome supervision, process supervision, and credit assignment

### 8.9.1 Outcome rewards

An outcome reward scores the final answer or terminal environment state:

```text
answer is correct?        → 1 / 0
unit tests pass?          → pass rate or binary pass
proof checker accepts?    → 1 / 0
user preferred response?  → learned scalar
```

It is cheap when a trusted verifier exists, but it provides little information about *why* the trajectory failed.

### 8.9.2 Process rewards

A process reward model scores intermediate reasoning steps. This can provide denser feedback and diagnose errors earlier. The central empirical reference is [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050), which compared process supervision and outcome supervision on mathematical reasoning and released PRM800K.

But process supervision is not a free upgrade:

- step labels are expensive or judge-dependent;
- natural-language rationales may not faithfully reflect latent computation;
- optimizing visible steps can favor plausible narration over correct internal search;
- a process reward model becomes another surface for exploitation.

### 8.9.3 Reward decomposition is a research design choice

For agent tasks, you may combine:

\[
r(\tau) =
\lambda_{\text{task}}r_{\text{terminal}}
+\lambda_{\text{format}}r_{\text{valid}}
+\lambda_{\text{safety}}r_{\text{policy}}
+\lambda_{\text{efficiency}}r_{\text{cost}}.
\]

Every added term changes the induced behavior. For example, rewarding short answers may suppress necessary reasoning; rewarding tool calls may create tool spam; rewarding a task test alone may create brittle overfit patches. Treat reward composition as a hypothesis, then red-team it.

---

## 8.10 RL with verifiable rewards (RLVR)

### 8.10.1 The promise

RLVR replaces a learned preference score with a checkable external signal:

- a mathematical answer matches a verifier;
- a program passes unit tests;
- a theorem checker accepts a proof;
- a SQL query returns the expected result;
- a tool trajectory reaches a target state;
- structured output validates against a schema plus semantic checker.

When the verifier has high precision, it can create massive feedback at lower cost than human comparison labels.

### 8.10.2 The essential caveat

“Verifiable” does **not** mean “perfectly aligned.” A verifier can be:

- incomplete;
- vulnerable to test leakage or test gaming;
- sparse;
- noisy due to environment nondeterminism;
- incorrectly scoped;
- correlated with shortcuts rather than the intended capability.

This caution is not philosophical decoration. Research has shown that apparent RLVR gains need careful interpretation, including work arguing that rewards weakly related to intended correctness can sometimes still yield improvements through distributional or selection effects. [Does RL Really Incentivize Reasoning?](https://arxiv.org/abs/2504.13837) · [Spurious Rewards](https://arxiv.org/abs/2506.10947)

### 8.10.3 GRPO: group-relative policy optimization

GRPO samples a group of \(G\) completions for the same prompt. If each completion receives reward \(r_i\), a simple group-normalized advantage is:

\[
A_i = \frac{r_i-\mu_r}{\sigma_r+\epsilon},
\quad
\mu_r=\frac{1}{G}\sum_{j=1}^G r_j.
\]

This answers a practical question: *which completion was better than the other attempts for this same problem?* It can avoid a separately trained value model, reducing some memory and systems complexity. The canonical open reference is [DeepSeekMath](https://arxiv.org/abs/2402.03300).

However, group normalization has real pathologies:

- if all rewards are equal, there is little learning signal;
- small groups may miss rare correct trajectories;
- very large groups are expensive;
- the group itself biases the estimated advantage;
- high reward variance does not automatically mean useful learning.

This is why current work studies prompt selection, group construction, replay, and rare-correct trajectories rather than treating GRPO as a final answer. [F-GRPO](https://arxiv.org/abs/2602.06717) · [Prioritized Replay for RL Post-training](https://arxiv.org/abs/2601.02648)

### 8.10.4 DeepSeek-R1: what to learn from it

The durable lesson of DeepSeek-R1 is not “pure RL magically creates intelligence.” The paper’s value is that it made an open, concrete case that large-scale RL with verifiable rewards can incentivize longer reasoning behavior and improve performance in verifiable domains, and that a pipeline can combine cold-start data, reasoning RL, rejection sampling, and later alignment stages. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

The correct research questions are:

1. What capabilities existed in the base policy’s support before RL?
2. Does RL create new solution families, redistribute probability mass toward rare successes, increase test-time search, or all three?
3. Which reward, sampling, and data-curriculum choices create the observed effect?
4. Does the gain transfer outside the verifier’s domain?
5. What is the compute cost per unit of robust capability?

These questions protect you from both hype and premature dismissal.

---

## 8.11 Modern reasoning RL: rollout strategy is part of the algorithm

A rollout is not just a data-collection step. In modern reasoning RL, it determines what evidence the optimizer ever sees.

### 8.11.1 Prompt selection and the learning frontier

If every sampled prompt is too easy, all trajectories receive reward 1 and the gradient vanishes. If every prompt is too hard, all receive reward 0 and the gradient also vanishes. Informative learning often occurs near the policy’s capability frontier, where some attempts succeed and some fail.

This connects LLM RL to curriculum learning, active learning, and prioritized replay. Current work studies rare-event prompts, success-rate-aware sampling, and replay policies precisely because **the prompt distribution controls gradient usefulness**. [Rare-Event Prompt Efficiency](https://arxiv.org/abs/2602.03452) · [Prioritized Replay](https://arxiv.org/abs/2601.02648)

### 8.11.2 Exploration versus exploitation

Sampling temperature, top-p, group size, tool budgets, and prompt diversity define the exploration policy. Low-entropy sampling may repeatedly produce the same mediocre trajectory. High-entropy sampling may waste budget in nonsense regions. The right setting changes through training.

A research-grade experiment logs:

- entropy and token-level KL;
- success distribution per prompt;
- number of unique trajectories;
- verifier pass rate versus sample count;
- length and tool-call distributions;
- policy-version age of every rollout.

### 8.11.3 Test-time scaling and train-time RL are coupled

When a model can sample multiple candidates, use a verifier, and select or revise, capability may rise without changing weights. RL can then train the model to produce better candidates, allocate reasoning budget, or use tools more effectively.

Do not conflate:

```text
pass@1 improvement      = better first sampled policy behavior
pass@k improvement      = better best-of-k opportunity / diversity
search improvement      = better selection or verification procedure
RL improvement          = changed model distribution after optimization
```

A careful paper must specify which one it measures.

### 8.11.4 Recent rollout research

A useful current map is [A Comprehensive Survey of Rollout Strategies for LLM Post-Training](https://arxiv.org/abs/2605.02913). Treat it as a bridge from optimizer-centric thinking to **data-generation-centric** thinking.

---

## 8.12 Agent reinforcement learning: from completions to environments

### 8.12.1 The environment changes everything

A coding or browser agent does not emit one answer and stop. It interacts:

```text
observe repository / browser / tool state
→ choose text or tool action
→ environment executes it
→ receive observation, error, file diff, webpage, or test result
→ continue until terminal success, failure, or budget exhaustion
```

This adds:

- long horizons;
- sparse terminal rewards;
- partially observable state;
- irreversible actions;
- tool errors and flaky environments;
- variable-length trajectories;
- high rollout cost;
- credit assignment across many actions.

### 8.12.2 Why plain RLVR struggles in agent settings

In a hard software task, a model may have a near-zero chance of producing a passing patch from scratch. Binary unit-test reward then gives almost all failures, which is a poor learning signal.

Agent-RLVR addresses this by adding guidance derived from plans, error feedback, and environment interactions, allowing guided attempts to create more learnable successful trajectories. In its reported software-engineering experiments, the authors use unit-test outcomes plus guidance and exploration to improve a Qwen-2.5-72B-Instruct agent on SWE-bench Verified. [Agent-RLVR](https://arxiv.org/abs/2506.11425)

The durable lesson is:

> For long-horizon agent RL, the environment, curriculum, guidance, and verifier are part of the learning algorithm.

### 8.12.3 A multi-turn agent return

For an agent trajectory with tool actions \(a_t\), observations \(o_t\), and potentially only terminal reward \(R_T\):

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R_T].
\]

This is mathematically simple and practically brutal. A terminal failure gives little information about whether the agent chose the wrong file, used the wrong command, misunderstood the task, failed a patch, or hit a sandbox issue. Useful agent-RL systems therefore study reward shaping, demonstrations, curriculum tasks, guided rollouts, replay, and trace analysis.

### 8.12.4 Tool use is an action-space design problem

An agent’s action interface defines the RL problem:

- coarse tools shrink horizon but may reduce control;
- primitive tools enlarge horizon but improve compositionality;
- ambiguous tool schemas inject action noise;
- unsafe tools require hard constraints, not merely negative rewards;
- a tool result can become an observation, a reward input, or both.

This is where Part VIII meets your Agentic Engineering curriculum. A better harness can make the RL environment more observable, safer, and easier to verify.

---

## 8.13 The real implementation: what a large-scale LLM RL system runs

### 8.13.1 The functional components

```text
                  ┌────────────────────────────────┐
                  │ Prompt / task sampler           │
                  │ curriculum + dataset versioning │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Rollout plane                                                               │
│ policy snapshot → generation servers → tool environments → trajectories     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Evaluation / reward plane                                                    │
│ verifier | reward model | generative judge | process model | safety checks  │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ trajectories + rewards + policy-version metadata
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Learner plane                                                               │
│ actor update | reference log-probs | critic (if used) | optimizer | shards  │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ checkpoint / new policy version
                └───────────────────────────────────────────────► rollout plane

Cross-cutting: experiment tracking, replay/audit store, evaluation suites,
policy/version governance, contamination controls, rollback, cost telemetry.
```

### 8.13.2 Why serving and RL training converge

Rollout generation looks like inference but differs from product serving:

- many candidates per prompt;
- long completions or tool traces;
- high sampling entropy;
- policy snapshots change frequently;
- log probabilities and token masks must be retained;
- tool environments and verifiers introduce variable latency;
- evaluation may require more expensive judge models.

A training stack therefore often combines a high-throughput inference engine with a distributed learner. Modern frameworks explicitly integrate components such as vLLM/SGLang, FSDP or Megatron-style training, Ray/SLURM orchestration, and task environments. [veRL](https://github.com/verl-project/verl) · [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) · [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html)

### 8.13.3 The policy-version problem

Every trajectory should record:

```text
policy_checkpoint_id
reference_checkpoint_id
sampling parameters
prompt/task dataset version
environment image/version
verifier version
reward-model / judge version
code commit
random seed when feasible
```

Without this, you cannot diagnose whether a reward jump came from a policy update, a changed test suite, a more permissive judge, a prompt shift, or a rollout runtime bug.

### 8.13.4 Synchronous and asynchronous training

**Synchronous on-policy RL** pauses or coordinates rollouts and updates so data remains close to the current policy. It is conceptually simple but may leave hardware idle.

**Asynchronous or off-policy RL** overlaps generation and learning, using slightly older samples. It can improve hardware utilization but creates policy-staleness and correction problems. Research on asynchronous RLHF studies this trade-off directly and reports that online DPO was relatively robust to off-policy data in its evaluated settings. [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)

### 8.13.5 Pseudocode: minimal GRPO-style verifier loop

```python
# Conceptual pseudocode. Not production-ready.
reference_policy = freeze(copy(policy))

for prompts in prompt_sampler:
    # 1. Roll out multiple attempts per prompt.
    groups = [
        policy.sample(prompt, n=group_size, temperature=temperature)
        for prompt in prompts
    ]

    # 2. Score each attempt with a deterministic verifier or environment.
    rewards = [
        [verifier(prompt, completion) for completion in completions]
        for prompt, completions in zip(prompts, groups)
    ]

    # 3. Normalize reward inside each prompt group.
    advantages = []
    for group_rewards in rewards:
        mu = mean(group_rewards)
        sigma = std(group_rewards)
        advantages.append([(r - mu) / (sigma + 1e-6) for r in group_rewards])

    # 4. Recompute token log-probabilities under the trainable policy.
    logp = policy.logprob(groups, prompts)
    ref_logp = reference_policy.logprob(groups, prompts)

    # 5. Optimize policy likelihood weighted by relative advantage,
    #    with a KL term that limits drift from the reference policy.
    loss = grpo_surrogate(logp, old_logp=logp.detach(), advantages=advantages)
    loss += beta * kl_penalty(logp, ref_logp)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), max_norm)
    optimizer.step()

    # 6. Run held-out evaluation, safety checks, and reward-hacking diagnostics.
    evaluate_and_maybe_rollback(policy)
```

The loop is small. The hard parts are the verifier, data curriculum, rollout infrastructure, versioning, numerical stability, and evaluation.

---

## 8.14 RL diagnostics: what to measure before believing a result

A single reward curve is not enough. At minimum, record:

| Category | Diagnostic |
|---|---|
| Policy movement | KL to reference, KL to rollout policy, entropy, token distribution shifts |
| Reward | mean, variance, quantiles, per-task success histogram, all-zero/all-one group rate |
| Learning signal | advantage distribution, effective group size, gradient norm, ratio clipping fraction |
| Behavior | answer length, reasoning length, tool-call count, refusal rate, format validity |
| Generalization | held-out in-domain, OOD, adversarial, and contamination-resistant evaluations |
| Verifier health | flaky-test rate, false positives/negatives, reward disagreement, exploit cases |
| Systems | generation tokens/sec, learner utilization, queue latency, policy lag, cost/solved task |
| Regression | base benchmark suite, safety, multilingual performance, calibration, code quality |

### 8.14.1 Reward hacking checklist

Ask after every apparent gain:

1. Could the model exploit the checker without solving the intended task?
2. Did it learn the benchmark’s formatting quirks?
3. Did response length or tool-call count change suspiciously?
4. Does the policy still work under paraphrases, hidden tests, new environments, and different seeds?
5. Do independent judges and humans agree with the reward?
6. Did a change in prompt distribution or verifier version create the improvement?

### 8.14.2 Evaluation is part of the objective

When you choose only one benchmark, you implicitly make it the shadow reward. A rigorous post-training project needs a matrix of capabilities and failure modes, not a leaderboard monoculture.

---

## 8.15 Failure modes and unresolved scientific questions

### 8.15.1 Reward misspecification and exploitation

A learned reward model is an approximation to preferences; a verifier is an approximation to task success. Optimization pressure discovers gaps. This is not an edge case. It is the default expectation in any sufficiently capable policy search process.

### 8.15.2 Do reasoning-RL gains reflect new reasoning?

There is active debate over whether RLVR creates fundamentally new reasoning behaviors or primarily reallocates probability toward rare successful behaviors already present in the base model’s distribution. Evidence differs by model, task, evaluation protocol, and sampling budget. Read opposing or cautionary work rather than adopting a team jersey. [Does RL Really Incentivize Reasoning?](https://arxiv.org/abs/2504.13837) · [RLVR Implicitly Incentivizes Correct Reasoning](https://arxiv.org/abs/2506.14245)

### 8.15.3 Credit assignment over long traces

Token-level policy gradients still struggle when one early decision determines a terminal outcome hundreds of tool actions later. Process rewards, value models, search, hindsight-style data, curriculum, and guided rollouts are all attempts to address this. None is a general cure.

### 8.15.4 Generalization beyond verifiable domains

Math and code provide structured evaluators. Medicine, science, economics, interpersonal dialogue, and open-ended research do not offer comparably clean ground truth. Work on expanding RLVR beyond classical verifiable domains is promising but makes evaluator validity the central problem. [Crossing the Reward Bridge](https://arxiv.org/abs/2503.23829)

### 8.15.5 Data and environment contamination

A benchmark win may reflect pretraining leakage, tool-environment leakage, prompt-template memorization, or contaminated tests. For agent environments, the issue expands to cached artifacts, package versions, network access, and state reuse.

### 8.15.6 Alignment tax and capability preservation

Post-training may improve target behavior while reducing other capabilities, calibration, diversity, or multilingual quality. The correct target is a Pareto frontier, not a single scalar.

### 8.15.7 Human oversight does not disappear

Replacing human labels with AI judges or verifiers changes where human labor lives: evaluator design, red-teaming, policy writing, test-authoring, data governance, and audit. It does not eliminate the need for legitimate objectives.

---

## 8.16 What changed from “RLHF” to modern LLM post-training?

| Earlier dominant picture | Modern broader picture |
|---|---|
| One reward model trained from pairwise human preferences | Multiple evaluator types: reward models, generative judges, process models, deterministic verifiers, tool environments |
| PPO as the visible default | PPO remains foundational, but DPO-family, GRPO-family, RLOO-like, asynchronous, and hybrid methods are common research choices |
| Single-turn text completion | long reasoning traces, multi-turn tools, coding agents, web agents, multimodal environments |
| Reward as the main learning bottleneck | prompt selection, rollout diversity, verifier quality, policy staleness, systems throughput, and evaluation all matter |
| RL runs after SFT | multi-stage pipelines can interleave distillation, SFT, RLVR, safety alignment, rejection sampling, and model merging |
| Training algorithm as the star | entire optimization **ecosystem** is the object of study |

This is the right mental transition for an AI engineering PhD: **an algorithmic acronym is only one component of a closed learning system.**

---

## 8.17 Canonical reading order for RL and post-training

Read in this order, writing a research card for each:

1. **Sutton & Barto**, Chapters 1–3, 5, 6, 9, 13. Learn MDPs, returns, value functions, TD learning, policy gradients, actor–critic.
2. **PPO**. Derive the probability ratio and clipping objective. [Paper](https://arxiv.org/abs/1707.06347)
3. **Deep RL from Human Preferences**. Learn why preferences can define rewards. [Paper](https://arxiv.org/abs/1706.03741)
4. **Fine-Tuning Language Models from Human Preferences** and **Learning to Summarize from Human Feedback**. Learn the direct historical bridge to language. [2019](https://arxiv.org/abs/1909.08593) · [2020](https://arxiv.org/abs/2009.01325)
5. **InstructGPT**. Diagram the full pipeline, including datasets and evaluation. [Paper](https://arxiv.org/abs/2203.02155)
6. **Constitutional AI**. Separate critique/revision from RLAIF/PPO. [Paper](https://arxiv.org/abs/2212.08073)
7. **DPO**. Derive the preference loss from the KL-regularized view. [Paper](https://arxiv.org/abs/2305.18290)
8. **Let’s Verify Step by Step**. Contrast process versus outcome supervision. [Paper](https://arxiv.org/abs/2305.20050)
9. **DeepSeekMath / GRPO**. Understand group-relative advantage and critic removal. [Paper](https://arxiv.org/abs/2402.03300)
10. **DeepSeek-R1** and **DAPO**. Separate reported recipe, infrastructure, and causal claims. [R1](https://arxiv.org/abs/2501.12948) · [DAPO](https://arxiv.org/abs/2503.14476)
11. **Agent-RLVR**. Move from completions to environments. [Paper](https://arxiv.org/abs/2506.11425)
12. **RLVR debate papers**. Read claims and caveats together. [Positive analysis](https://arxiv.org/abs/2506.14245) · [Cautionary analysis](https://arxiv.org/abs/2504.13837) · [Spurious rewards](https://arxiv.org/abs/2506.10947)
13. **System papers and framework source.** [HybridFlow](https://arxiv.org/abs/2409.19256) · [veRL](https://github.com/verl-project/verl) · [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) · [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html)

---

# Part VIII-A — RL implementation laboratory and research agenda

## 8A.1 The implementation ladder: build understanding in the right order

Do not start with a 70B checkpoint and a trendy acronym. Build one concept at a time.

### Lab 1 — Multi-armed bandit and REINFORCE

**Build:** A two- or ten-armed bandit in NumPy/PyTorch. Implement a softmax policy and REINFORCE.

**Measure:** reward mean, action probabilities, gradient variance, effect of a moving-average baseline.

**Understand:** why sampling is necessary, why a baseline changes variance but not expected gradient, and why reward scale changes optimization behavior.

### Lab 2 — Contextual bandit as one-step language alignment

**Build:** Prompts map to a small set of response templates. A known reward function marks a response appropriate or inappropriate.

**Measure:** learned policy per prompt, entropy collapse, effect of KL to a reference policy.

**Understand:** the simplest analogue of one-turn reward optimization.

### Lab 3 — Sequence REINFORCE on verifiable arithmetic

**Build:** A tiny character- or token-level model produces arithmetic expressions or answers. Reward exact correctness.

**Measure:** pass@1, pass@k, reward variance, sequence length, policy entropy.

**Understand:** sequence credit assignment and sparse terminal reward.

### Lab 4 — Preference data and a reward model

**Build:** Generate response pairs, label them with a synthetic but imperfect preference rule, train a Bradley–Terry reward model.

**Measure:** pairwise accuracy, calibration, out-of-distribution ranking, adversarial reward exploit examples.

**Understand:** why “high reward-model accuracy” does not certify safe optimization.

### Lab 5 — DPO from scratch on a small preference set

**Build:** Implement the DPO loss directly from completion log-probabilities and a frozen reference model.

**Measure:** chosen/rejected margin, reference KL, helpfulness-style task metrics, length shifts.

**Understand:** offline preference optimization and the role of \(\beta\).

### Lab 6 — PPO RLHF on a tiny open model

**Build:** An SFT policy, reward model, reference policy, optional value head, rollout generation, PPO update.

**Measure:** clipped-ratio fraction, KL, value loss, entropy, reward, held-out quality.

**Understand:** why PPO is more than “use reward and backprop.”

### Lab 7 — GRPO / RLVR on a verifier

**Build:** For each prompt, sample groups of completions; verify with exact answer or unit tests; normalize group rewards; optimize the policy.

**Measure:** all-zero / all-one group fraction, group-size sensitivity, rare-correct rate, verifier exploit attempts.

**Understand:** why prompt difficulty and rollout policy determine learnability.

### Lab 8 — Multi-turn agent RL in a sandbox

**Build:** A small coding or database environment with a fixed tool suite, deterministic reset, explicit time/tool budgets, and hidden tests.

**Measure:** success rate, trajectory length, tool schema errors, action entropy, hidden-test generalization, cost per solved task.

**Understand:** the environment is part of the objective.

### Lab 9 — Asynchronous rollout/learner experiment

**Build:** Separate rollout workers from learner workers. Tag every trajectory with policy version and impose controlled staleness.

**Measure:** hardware utilization, queue delay, policy lag, final quality, and degradation as staleness grows.

**Understand:** why a systems optimization changes the statistical training regime.

### Lab 10 — Reproduce a claim, then falsify it

Choose one small published claim: e.g., group size effect, KL coefficient effect, process reward effect, or reference-free preference objective. Reproduce a constrained version, then change a hidden assumption deliberately.

**Deliverable:** a paper-style report with hypothesis, setup, ablation table, negative results, failure analysis, and reproducibility manifest.

---

## 8A.2 Open-source systems to learn from

| System | Best use | What to inspect |
|---|---|---|
| [Hugging Face TRL](https://huggingface.co/docs/trl/) | educational baseline and fast experiments | trainer API, DPO/GRPO/PPO implementations, dataset contracts |
| [veRL](https://github.com/verl-project/verl) | scalable RL post-training research | hybrid controller, rollout/training integration, FSDP/Megatron/vLLM interfaces |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | accessible distributed RLHF / agent RL | Ray orchestration, async examples, inference integration |
| [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | large-scale and multimodal post-training | distributed configuration, environment abstractions, Megatron integration |
| [Open-Instruct](https://github.com/allenai/open-instruct) | transparent instruction/post-training experiments | data recipes, evaluation, asynchronous-RL research code |

Read framework source only after you can implement the small version. Otherwise a distributed codebase becomes a fog machine with CUDA graphs.

---

## 8A.3 A PhD-quality RL experiment protocol

Before running a job, write a pre-registration-style experiment sheet:

```text
Question:
Does [intervention] improve [capability] under [budget] without harming [held-out capabilities]?

Hypothesis:
Mechanism predicted by the theory.

Base policy and initialization:
Checkpoint, SFT data, tokenizer, precision, seed policy.

Environment:
Prompt dataset/version, tools, container image, network rules, verifier version.

Intervention:
Exact objective, coefficients, sampling parameters, group size, rollout horizon,
policy-update schedule, and whether data are on-policy.

Baselines:
At minimum: SFT/reference, no-intervention control, and closest simpler baseline.

Metrics:
Quality, generalization, reward, KL, entropy, length, failure modes, cost, throughput.

Ablations:
Remove the alleged mechanism. Vary one important scale/budget parameter.

Stopping / rollback rules:
Safety, regression, instability, or cost ceilings.

Reproducibility artifacts:
Code commit, configs, checkpoint IDs, dataset hashes, environment image, logs.
```

This discipline will make your experiments interpretable before it makes them impressive.

---

## 8A.4 Research questions worth pursuing

1. **Reward validity:** How can we estimate whether a reward or judge continues to represent the intended goal after the policy distribution shifts?
2. **Verifier robustness:** How do we construct verifiers that are cheap, hard to game, and meaningful outside benchmark templates?
3. **Long-horizon credit assignment:** Can process models, causal trace attribution, counterfactual rollouts, or learned world models provide useful credit without optimizing superficial reasoning prose?
4. **Efficient exploration:** Which prompt distributions, group sizes, and sampling strategies maximize learning signal per generated token?
5. **Online/offline trade-off:** When does stale data harm a policy update enough to erase asynchronous systems gains?
6. **Agent-environment co-design:** Which tool abstractions and state representations make an agent both safer and easier to train?
7. **Capability preservation:** How can post-training improve one task family without collapsing calibration, multilingual behavior, safety, or broad knowledge?
8. **Reward model transparency:** Are generative / reasoning reward models more reliable than scalar heads, and under what adversarial conditions?
9. **Multi-agent RL:** How should credit and reward be assigned when a planner, coder, reviewer, and tool executor collectively solve a task?
10. **Evaluation science:** What benchmark design can distinguish policy improvement from test-time search, contamination, formatting exploitation, or increased answer length?

---

## 8A.5 The current frontier, June 2026

The productive frontier is not a single “best RL algorithm.” It is the joint optimization of:

\[
\text{policy} + \text{rollout distribution} + \text{verifier/judge} + \text{environment} + \text{systems throughput} + \text{evaluation}.
\]

The recent literature contains several live strands:

- **Rollout strategy and prompt efficiency:** how to find tasks with non-degenerate learning signal. [Rollout survey](https://arxiv.org/abs/2605.02913) · [Prioritized replay](https://arxiv.org/abs/2601.02648)
- **GRPO/group-statistics corrections:** how group sampling misses rare correct trajectories and biases optimization. [F-GRPO](https://arxiv.org/abs/2602.06717)
- **Asynchronous and off-policy post-training:** how to overlap expensive generation and learning while controlling stale-policy error. [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)
- **Agent training:** how to make sparse multi-step environments learnable using guidance, tools, and environment feedback. [Agent-RLVR](https://arxiv.org/abs/2506.11425)
- **Self-feedback and intrinsic signals:** whether calibrated model uncertainty or self-critique can provide useful feedback without external labels. [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590) · [RLSF](https://arxiv.org/abs/2507.21931)
- **Reward-model reasoning:** whether judges that deliberate can provide higher-quality optimization targets. [RM-R1](https://arxiv.org/abs/2505.02387)
- **Multimodal and environment-rich RL:** applying post-training beyond text-only completions. [EasyR1](https://github.com/hiyouga/EasyR1) · [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html)

Treat every item above as a hypothesis space, not a finished answer.



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

## Project 9 — Build the progressive RL post-training laboratory

**Goal:** Learn RL through increasingly realistic closed loops rather than jumping directly to a large framework.

- Start with a bandit and REINFORCE baseline.
- Implement a preference reward model and DPO on a small paired dataset.
- Build PPO with a toy reward model, reference policy, and KL monitoring.
- Build verifier-based group rollouts for arithmetic or unit-test coding tasks.
- Add a deterministic multi-turn sandbox with tools and hidden tests.
- Track policy version, entropy, KL, reward distribution, all-zero groups, reward hacking, answer length, tool use, hidden-test success, and cost per solved task.

**You learn:** RL is a learning system made of policy, rollout distribution, evaluator, environment, distributed runtime, and evaluation, not merely a loss function.

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

# Part XI — A 24-week engineering foundation

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

**Week 22:** RL foundations: MDPs/POMDPs, REINFORCE, baselines, actor–critic, PPO, policy staleness. Build a bandit and sequence-RL toy implementation.

**Week 23:** LLM post-training history and implementation: SFT, reward modeling, RLHF/PPO, Constitutional AI/RLAIF, DPO-family. Implement DPO and a small reward model.

**Week 24:** RLVR, GRPO, agent RL, rollout systems, reward hacking, and evaluation. Build the first verifier-based group rollout loop and write a failure analysis.

## Phase 7 — Research synthesis and capstone (Weeks 25–26, optional extension)

**Week 25:** Read the frontier papers in Part IX and Part VIII-A. Write a “bottleneck transfer” memo for each: What cost moved? What new failure mode emerged? What experiment would falsify the claim?

**Week 26:** Build the inference-aware agent platform capstone. Publish a reproducible benchmark report with model/version, prompt mix, hardware, precision, runtime, cache policy, batch policy, quality metrics, and cost estimates.

---

# Part XI-A — A 12-month PhD-style research path

> **Purpose:** The 24-week path gives you an engineering foundation. This 12-month program is for building the habits of a research engineer: derivation, replication, systems measurement, falsification, and original problem selection. It is intentionally demanding.

> **Mathematics sequencing note:** The detailed mathematics-only calendar now lives in [Part I-A](#part-i-a--mathematics-research-track-a-bounded-12-month-curriculum). Use it as a dedicated first year, or run it in parallel at half pace while following this broader systems path. The earlier “Months 1–3: mathematics” line below is therefore a summary, not the full mathematics syllabus.

## Quarter 1 — Mathematics, language modeling, and experimental discipline

**Months 1–3:**

- Follow the detailed [Part I-A mathematics calendar](#part-i-a--mathematics-research-track-a-bounded-12-month-curriculum) at the appropriate pace; this three-month summary is only a broad integration target.
- Implement a small GPT, train it, profile it, and write a model card plus reproducibility manifest.
- Reproduce one canonical result each for attention/KV caching and mixed precision.
- Maintain research cards for 12 foundational papers.
- Deliverable: a 10–15 page technical note deriving the Transformer forward/backward pass and measuring the compute/memory trade-offs in your implementation.

## Quarter 2 — Architectures, compilers, hardware, and distributed training

**Months 4–6:**

- Implement and compare dense MLP, toy MoE, attention, and a simplified state-space layer on controlled tasks.
- Study GPU memory hierarchy, roofline analysis, CUDA/Triton kernels, and distributed training parallelisms.
- Reproduce one systems result: FlashAttention-style tiling, a PagedAttention-like cache allocator, or a small prefill/decode scheduler.
- Deliverable: a systems paper replication report with profiling traces, roofline reasoning, and a negative-result section.

## Quarter 3 — Inference, quantization, and model-runtime co-design

**Months 7–9:**

- Benchmark at least two runtimes under realistic request mixes, including long prompts, repeated prefixes, structured output, and concurrency.
- Conduct a quantization study across weights, activations, or KV cache, evaluating quality *and* TTFT/TPOT/goodput.
- Investigate one architecture/runtime interaction: GQA versus KV cache, MoE routing versus batching, quantization versus speculative decoding, or long context versus cache eviction.
- Deliverable: a reproducible serving benchmark suite and a workshop-quality memo explaining a bottleneck transfer.

## Quarter 4 — Post-training, RL, agents, and original research

**Months 10–12:**

- Build the labs in Part VIII-A through a GRPO/RLVR system, then a deterministic multi-turn agent environment.
- Reproduce one result from DPO, PPO/RLHF, GRPO/RLVR, or Agent-RLVR at reduced scale.
- Run one planned falsification experiment, not merely a hyperparameter sweep.
- Write an original proposal around verifier quality, rollout efficiency, agent-environment design, or capability-preserving post-training.
- Deliverable: a research-style paper with abstract, related work, method, experimental protocol, results, limitations, and release artifact.

## Weekly cadence for all 12 months

```text
2 sessions  → derivation / textbooks
2 sessions  → implementation / debugging
1 session   → paper reading and research cards
1 session   → experiment analysis / writing
1 session   → open exploration, replication, or rest
```

The weekly unit is not “finish a paper.” It is “form and test a precise understanding.”

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

## Reinforcement learning, alignment, and post-training

52. [Sutton & Barto: Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)
53. [REINFORCE / Williams 1992](https://link.springer.com/article/10.1007/BF00992696)
54. [Q-learning / Watkins & Dayan 1992](https://link.springer.com/article/10.1007/BF00992698)
55. [TRPO](https://arxiv.org/abs/1502.05477)
56. [PPO](https://arxiv.org/abs/1707.06347)
57. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
58. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
59. [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
60. [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155)
61. [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
62. [Constitutional AI / RLAIF](https://arxiv.org/abs/2212.08073)
63. [DPO](https://arxiv.org/abs/2305.18290)
64. [Let’s Verify Step by Step / PRM800K](https://arxiv.org/abs/2305.20050)
65. [KTO](https://arxiv.org/abs/2402.01306)
66. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
67. [ORPO](https://arxiv.org/abs/2403.07691)
68. [SimPO](https://arxiv.org/abs/2405.14734)
69. [Direct Reward Optimisation / Offline Regularised RL](https://arxiv.org/abs/2405.19107)
70. [OpenRLHF paper](https://arxiv.org/abs/2405.11143)
71. [HybridFlow](https://arxiv.org/abs/2409.19256)
72. [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)
73. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
74. [DAPO](https://arxiv.org/abs/2503.14476)
75. [Crossing the Reward Bridge](https://arxiv.org/abs/2503.23829)
76. [Does RL Really Incentivize Reasoning?](https://arxiv.org/abs/2504.13837)
77. [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)
78. [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590)
79. [Agent-RLVR](https://arxiv.org/abs/2506.11425)
80. [RLVR and Correct Reasoning](https://arxiv.org/abs/2506.14245)
81. [Spurious Rewards](https://arxiv.org/abs/2506.10947)
82. [Reinforcement Learning for LLM Post-Training: A Survey](https://arxiv.org/abs/2407.16216)
83. [A Survey of RL for Large Reasoning Models](https://arxiv.org/abs/2509.08827)
84. [A Comprehensive Survey of Rollout Strategies for LLM Post-Training](https://arxiv.org/abs/2605.02913)
85. [F-GRPO](https://arxiv.org/abs/2602.06717)
86. [Prioritized Replay for RL Post-training](https://arxiv.org/abs/2601.02648)

## Open-source post-training systems

87. [Hugging Face TRL documentation](https://huggingface.co/docs/trl/)
88. [veRL](https://github.com/verl-project/verl)
89. [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
90. [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html)
91. [Open-Instruct](https://github.com/allenai/open-instruct)

---

# A note on currency

This guide was assembled with sources checked through **June 23, 2026**. Research papers in Part VIII and Part IX are snapshots of an active field; where papers disagree, the guide intentionally preserves the disagreement rather than presenting a premature consensus. Architecture and math fundamentals change slowly; model releases, hardware specifications, runtime APIs, benchmark numbers, and vendor support matrices change quickly.

Before relying on a fast-moving claim, check:

- the paper’s latest version and released code;
- the model card / technical report;
- runtime release notes for vLLM, SGLang, TensorRT-LLM, Neuron, or the relevant stack;
- accelerator vendor documentation for exact precision, memory, and interconnect support;
- benchmark methodology: model version, prompt/output lengths, batch size, hardware, runtime, precision, and metric definition.

> **Final engineering rule:** Treat every claimed speedup as a conditional statement. Ask: *on which model, on which hardware, at which precision, under which workload, compared with which baseline, while preserving which quality metric?*

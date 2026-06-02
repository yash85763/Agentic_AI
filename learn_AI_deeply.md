Below is a complete book-based roadmap to learn how LLMs are made, with strong math, deep learning, transformers, Mixture-of-Experts models like Mixtral, reinforcement learning, and RL for LLMs.

Think of this as a multi-volume curriculum, not a random reading list.

⸻

The Big Sequence

0. Mathematical thinking
1. Linear algebra
2. Calculus
3. Probability
4. Statistics
5. Optimization
6. Information theory
7. Classical machine learning
8. Deep learning
9. NLP and transformers
10. Build an LLM from scratch
11. Reinforcement learning
12. Deep reinforcement learning
13. RLHF / DPO / GRPO / reasoning models
14. Systems: GPUs, distributed training, kernels, inference
15. Research papers

Your goal should be:

First understand the mathematics of learning, then understand neural networks, then transformers, then LLM training, then RL/RLHF, then modern architectures like MoE, Mixtral, DeepSeek-style reasoning models, and agentic systems.

⸻

Volume 0: Mathematical Maturity

This phase teaches you how to read mathematical arguments without fear.

1. How to Prove It — Daniel J. Velleman

Read this first.

Focus on:

Logic
Sets
Functions
Relations
Proofs
Induction
Mathematical notation

Why it matters for LLMs:

LLM papers often assume you can follow abstract notation. This book helps you stop being blocked by symbols.

Read: Chapters 1–6.

⸻

Volume 1: Linear Algebra

Linear algebra is the language of embeddings, attention, neural networks, LoRA, quantization, SVD, PCA, and GPU matrix multiplication.

2. Introduction to Linear Algebra — Gilbert Strang

This should be your first serious AI math book.

Focus on:

Vectors
Matrices
Matrix multiplication
Column space
Null space
Rank
Orthogonality
Projections
Eigenvalues
Eigenvectors
SVD

Why it matters:

tokens → embeddings → matrix multiplications → attention → transformer blocks

LLMs are mostly repeated matrix operations.

Read: Almost the whole book, but especially chapters on vector spaces, orthogonality, eigenvalues, and SVD.

⸻

3. Linear Algebra Done Right — Sheldon Axler

This is more abstract and deeper.

Do not read it before Strang.

Focus on:

Vector spaces
Linear maps
Invariant subspaces
Eigenvalues
Spectral theorem
Inner product spaces

Why it matters:

This gives you the deeper intuition behind representation spaces and transformations.

Read selectively after Strang.

⸻

Volume 2: Calculus

Calculus explains how models learn through gradients.

4. Calculus — Michael Spivak

This is rigorous and beautiful.

Focus on:

Limits
Derivatives
Chain rule
Taylor expansion
Integrals
Sequences
Series

Why it matters:

Backpropagation is essentially the chain rule applied repeatedly through a computation graph.

Read selectively. You do not need to master every proof at first.

⸻

5. Vector Calculus — Marsden and Tromba

After single-variable calculus, move to multivariable calculus.

Focus on:

Partial derivatives
Gradients
Jacobians
Hessians
Directional derivatives
Multivariable chain rule

Why it matters:

Neural network parameters are not one variable. They are millions or billions of variables. Training is multivariable optimization.

⸻

Volume 3: Probability

LLMs are probability models.

They do not “know” the next word. They estimate:

P(next token | previous tokens)

6. Introduction to Probability — Bertsekas and Tsitsiklis

Focus on:

Conditional probability
Bayes rule
Random variables
Expectation
Variance
Distributions
Law of large numbers
Central limit theorem

Why it matters:

Next-token prediction, sampling, temperature, top-k, top-p, RL policies, and uncertainty all depend on probability.

⸻

7. Probability Theory: The Logic of Science — E.T. Jaynes

Read this later, not first.

Why it matters:

Jaynes teaches probability as reasoning under uncertainty. That mindset is very useful for understanding intelligence, inference, and language models.

⸻

Volume 4: Statistics

Statistics teaches you how models estimate patterns from data.

8. All of Statistics — Larry Wasserman

Focus on:

MLE
Bias and variance
Estimation
Confidence intervals
Regression
Bayesian inference
Hypothesis testing
Nonparametric methods

Why it matters:

Training an LLM is statistical estimation at massive scale.

The model sees data and adjusts parameters so its predicted distribution becomes closer to the real distribution.

⸻

Volume 5: Optimization

Optimization is where many AI engineers stay weak.

But training LLMs is fundamentally an optimization problem.

9. Convex Optimization — Stephen Boyd and Lieven Vandenberghe

Focus on:

Convex sets
Convex functions
Gradient descent
Lagrange multipliers
Duality
Constrained optimization

Why it matters:

Deep learning is mostly non-convex, but convex optimization gives you the language to understand why optimization is hard and what good optimization should look like.

⸻

10. Numerical Optimization — Jorge Nocedal and Stephen Wright

Read this after Boyd.

Focus on:

Gradient descent
Newton methods
Quasi-Newton methods
Line search
Trust regions
Stochastic optimization

Why it matters:

Adam, SGD, momentum, learning-rate schedules, and training stability are all optimization topics.

⸻

Volume 6: Information Theory

Information theory explains entropy, cross-entropy loss, KL divergence, compression, and why language modeling works.

11. Elements of Information Theory — Cover and Thomas

Focus on:

Entropy
Cross entropy
KL divergence
Mutual information
Source coding
Channel capacity
Compression

Why it matters:

The basic LLM objective is usually cross-entropy loss.

In simple terms:

The model predicts a probability distribution.
The true next token is known.
Cross-entropy punishes the model when it assigns low probability to the correct token.

This book explains the deeper theory behind that.

⸻

Volume 7: Classical Machine Learning

Before deep learning, understand what machine learning was trying to solve.

12. Pattern Recognition and Machine Learning — Christopher Bishop

Focus on:

Linear models
Bayesian methods
Neural networks
Kernel methods
Graphical models
EM algorithm
Mixture models
Approximate inference

Why it matters:

Many modern AI concepts have roots here: representation learning, probabilistic modeling, latent variables, mixture models, and approximate inference.

⸻

13. The Elements of Statistical Learning — Hastie, Tibshirani, Friedman

This is mathematically deeper than most ML books.

Focus on:

Bias-variance tradeoff
Regularization
Decision trees
Boosting
SVMs
Model selection
Ensemble methods

Why it matters:

It helps you understand generalization, overfitting, and why scaling data/models changes behavior.

⸻

Volume 8: Deep Learning

Now you are ready for neural networks properly.

14. Understanding Deep Learning — Simon J.D. Prince

Use this before Goodfellow.

It is more modern and more readable. MIT Press describes it as an accessible and up-to-date treatment that balances theory and practice.  

Focus on:

Neural networks
Loss functions
Backpropagation
Optimization
Regularization
Convolutional networks
Transformers
Generative models

Why it matters:

This book builds intuition very well.

⸻

15. Deep Learning — Ian Goodfellow, Yoshua Bengio, Aaron Courville

This is the classic deep learning textbook. The official site describes it as a resource for students and practitioners entering machine learning and deep learning, and the online version is available for free.  

Focus on:

Linear algebra review
Probability review
Numerical computation
Machine learning basics
Deep feedforward networks
Regularization
Optimization
CNNs
RNNs
Representation learning
Generative models

Why it matters:

This is the bridge from classical ML to modern neural networks.

Read after Prince, not before.

⸻

Volume 9: NLP and Transformers

Now move into language.

16. Speech and Language Processing — Jurafsky and Martin

Focus on:

Language modeling
N-grams
POS tagging
Parsing
Semantics
Information extraction
Neural language models
Transformers
Machine translation
Question answering
Dialogue systems

Why it matters:

Before transformers, NLP had decades of ideas. This book helps you understand what transformers replaced and what they inherited.

⸻

17. Natural Language Processing with Transformers — Lewis Tunstall, Leandro von Werra, Thomas Wolf

Focus on:

Hugging Face ecosystem
Tokenizers
Transformers
Fine-tuning
Text classification
Question answering
Summarization
Generation
Efficient deployment

Why it matters:

This is practical. It shows how transformer models are actually used.

⸻

Volume 10: Build an LLM From Scratch

This is where your understanding becomes real.

18. Build a Large Language Model (From Scratch) — Sebastian Raschka

This should be one of your most important books.

The author’s page describes it as a step-by-step guide to creating your own LLM, explaining each stage with diagrams, text, and examples.   The companion page says it covers text embeddings, attention, GPT-style architecture components, pretraining, and finetuning.  

Focus on:

Tokenization
Token embeddings
Positional embeddings
Self-attention
Causal attention
Multi-head attention
Transformer blocks
GPT architecture
Pretraining
Instruction fine-tuning
Classification fine-tuning

Why it matters:

This book makes LLMs feel buildable instead of magical.

Read this with code.

Do not just read. Implement.

⸻

Volume 11: Reinforcement Learning Foundations

Now start RL.

RL is not just another ML topic. It is about learning through interaction, reward, and long-term consequences.

19. Reinforcement Learning: An Introduction — Richard Sutton and Andrew Barto

This is the RL bible. The official page lists the second edition from MIT Press, and the authors provide the full book online.  

Focus on:

Multi-armed bandits
Markov Decision Processes
Bellman equations
Dynamic programming
Monte Carlo methods
Temporal-difference learning
SARSA
Q-learning
Eligibility traces
Policy gradients
Actor-critic methods

Why it matters:

RLHF, PPO, reward models, reasoning models, and agents all depend on these foundations.

Read slowly.

This book is not optional.

⸻

20. Algorithms for Decision Making — Mykel Kochenderfer, Tim Wheeler, Kyle Wray

Read this alongside Sutton and Barto.

Focus on:

Sequential decision making
MDPs
POMDPs
Planning
Bandits
Reinforcement learning
Policy search
Multi-agent decision making

Why it matters:

It gives you a broader decision-making view, not just RL algorithms.

⸻

21. Algorithms for Reinforcement Learning — Csaba Szepesvári

Read after Sutton and Barto.

Focus on:

Value prediction
Control
Planning
Approximation
Policy gradient methods
Theoretical guarantees

Why it matters:

This book is more mathematical and compact. It helps you move from “I know RL” to “I understand RL derivations.”

⸻

Volume 12: Deep Reinforcement Learning

Now combine neural networks with RL.

22. Deep Reinforcement Learning Hands-On — Maxim Lapan

Focus on:

DQN
Double DQN
Dueling DQN
Policy gradients
A2C
A3C
PPO
DDPG
SAC
Model-based RL

Why it matters:

You need to implement RL algorithms, not just read equations.

This book is code-heavy, which is good.

⸻

23. Grokking Deep Reinforcement Learning — Miguel Morales

Read this if Sutton and Barto feels too abstract.

Focus on:

Intuition
MDPs
Value methods
Policy methods
Actor-critic
Deep Q-learning
Policy gradients

Why it matters:

It gives intuitive explanations before heavy theory.

⸻

Volume 13: RL for LLMs

There are not many perfect “books” here yet. This area is paper-heavy. But you should study it after RL foundations.

You need to understand this pipeline:

Pretraining
        ↓
Supervised fine-tuning
        ↓
Preference data
        ↓
Reward model
        ↓
RLHF / PPO
        ↓
DPO / ORPO / GRPO / verifier-based training

Book-adjacent resources to use here

24. Hugging Face course/materials on RLHF and alignment

Use these after you know transformers and basic RL.

Focus on:

Reward modeling
Preference datasets
PPO for language models
KL penalty
Reference model
Policy model
DPO
Evaluation

⸻

Papers to read after the RL books

Read these in this order:

1. InstructGPT / Training language models to follow instructions with human feedback
2. Learning to summarize from human feedback
3. Direct Preference Optimization
4. Constitutional AI
5. RLAIF papers
6. ORPO
7. DeepSeekMath / GRPO-related work
8. DeepSeek-R1

The key idea:

Traditional RL:

state = environment observation
action = move
reward = environment score

LLM RL:

state = prompt/context
action = generated response/tokens
reward = human preference, reward model score, verifier score, or task correctness

⸻

Volume 14: Mixture of Experts, Mixtral, and Advanced LLM Architectures

For Mixtral-like models, you need three foundations:

Transformers
Sparse computation
Mixture-of-Experts routing
Distributed training

There is no single perfect book for MoE LLMs, so use papers after the transformer books.

Books that help before MoE papers

25. Designing Machine Learning Systems — Chip Huyen

Focus on:

ML systems
Data distribution
Training-serving skew
Monitoring
Deployment
Continual learning
ML pipelines

Why it matters:

LLMs are not just models. They are systems.

⸻

26. Machine Learning Systems — Jeff Smith / Chip Huyen-style system resources

Use ML systems resources to understand how models are trained, deployed, monitored, and scaled.

⸻

MoE papers to read

Read in this order:

1. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
2. GShard
3. Switch Transformers
4. GLaM
5. Mixtral of Experts
6. DeepSeek-V2 / DeepSeek-V3 technical reports

What you are trying to understand:

Dense model:
Every token uses every parameter.
MoE model:
Each token is routed to only a few experts.
Example:
Mixtral 8x7B may have many expert parameters,
but each token activates only a subset.

Core questions:

How does the router choose experts?
How is load balancing handled?
Why does sparse activation reduce compute?
Why is training MoE unstable?
How does distributed expert parallelism work?

⸻

Volume 15: Systems, GPUs, Kernels, and Distributed Training

To really understand modern LLMs, you must eventually learn systems.

27. Computer Systems: A Programmer’s Perspective — Bryant and O’Hallaron

Focus on:

Memory hierarchy
Caching
Linking
Processes
Virtual memory
Concurrency

Why it matters:

LLM inference speed depends heavily on memory movement, not just math.

⸻

28. Programming Massively Parallel Processors — Hwu, Kirk, El Hajj

Focus on:

GPU architecture
CUDA programming
Threads
Blocks
Warps
Memory coalescing
Shared memory
Parallel reductions
Matrix multiplication

Why it matters:

Training and inference are GPU problems.

Attention, matmul, FlashAttention, quantization kernels, and batching only make sense when you understand GPUs.

⸻

29. Distributed Machine Learning Patterns — Yuan Tang

Focus on:

Data parallelism
Model parallelism
Pipeline parallelism
Parameter servers
Distributed training
Fault tolerance

Why it matters:

Large LLMs cannot be trained on one GPU.

You need concepts like:

Data parallelism
Tensor parallelism
Pipeline parallelism
Expert parallelism
ZeRO
FSDP
Checkpointing

⸻

The Actual Reading Order

Here is the clean sequence I recommend.

Stage 1: Core Math Foundation

Read in this order:

1. How to Prove It — Velleman
2. Introduction to Linear Algebra — Strang
3. Calculus — Spivak
4. Vector Calculus — Marsden & Tromba
5. Introduction to Probability — Bertsekas & Tsitsiklis
6. All of Statistics — Wasserman

Outcome:

You can understand gradients, probability distributions, loss functions, and model training.

⸻

Stage 2: Deeper Math for AI

7. Convex Optimization — Boyd & Vandenberghe
8. Elements of Information Theory — Cover & Thomas
9. Numerical Optimization — Nocedal & Wright
10. Linear Algebra Done Right — Axler

Outcome:

You understand optimization, entropy, KL divergence, numerical stability, and deeper vector-space reasoning.

⸻

Stage 3: Machine Learning

11. Pattern Recognition and Machine Learning — Bishop
12. Elements of Statistical Learning — Hastie, Tibshirani, Friedman

Outcome:

You understand ML before deep learning.

⸻

Stage 4: Deep Learning

13. Understanding Deep Learning — Simon Prince
14. Deep Learning — Goodfellow, Bengio, Courville

Outcome:

You understand neural networks, backpropagation, regularization, optimization, and representation learning.

⸻

Stage 5: NLP and Transformers

15. Speech and Language Processing — Jurafsky & Martin
16. Natural Language Processing with Transformers — Tunstall, von Werra, Wolf
17. Build a Large Language Model From Scratch — Sebastian Raschka

Outcome:

You can build a small GPT-like model and understand tokenization, embeddings, attention, transformer blocks, pretraining, and fine-tuning.

⸻

Stage 6: Reinforcement Learning

18. Reinforcement Learning: An Introduction — Sutton & Barto
19. Algorithms for Decision Making — Kochenderfer, Wheeler, Wray
20. Algorithms for Reinforcement Learning — Szepesvári
21. Deep Reinforcement Learning Hands-On — Maxim Lapan

Outcome:

You understand MDPs, Bellman equations, Q-learning, policy gradients, actor-critic, PPO, and deep RL.

⸻

Stage 7: LLM Post-Training and Alignment

No single perfect book exists yet. Use papers and practical resources.

Read:

22. InstructGPT paper
23. Learning to Summarize from Human Feedback
24. Constitutional AI
25. Direct Preference Optimization
26. ORPO
27. DeepSeek-R1
28. DeepSeek-V3 technical report

Outcome:

You understand RLHF, reward models, PPO, DPO, ORPO, GRPO-style reasoning training, and verifier-based learning.

⸻

Stage 8: MoE, Mixtral, and Modern Architectures

Read:

29. Outrageously Large Neural Networks
30. GShard
31. Switch Transformers
32. GLaM
33. Mixtral of Experts
34. DeepSeek-V2 / V3 reports

Outcome:

You understand sparse expert routing, load balancing, expert parallelism, and why MoE models are powerful.

⸻

Stage 9: Systems

35. Computer Systems: A Programmer’s Perspective
36. Programming Massively Parallel Processors
37. Distributed Machine Learning Patterns

Outcome:

You understand GPU memory, kernels, CUDA, distributed training, parallelism, inference bottlenecks, and why serving LLMs is a systems problem.

⸻

A Practical 18-Month Plan

Months 1–3: Math Core

How to Prove It
Strang Linear Algebra
Bertsekas Probability
Basic calculus review

Main goal:

Understand vectors, matrices, probability, derivatives.

Build small notebooks:

Matrix multiplication from scratch
Gradient descent from scratch
Softmax from scratch
Cross-entropy from scratch

⸻

Months 4–6: Optimization + Statistics + Information Theory

All of Statistics
Boyd Convex Optimization
Cover & Thomas selected chapters

Main goal:

Understand why models minimize loss and how learning is statistical estimation.

Build:

Linear regression from scratch
Logistic regression from scratch
Tiny neural network from scratch
Backpropagation manually

⸻

Months 7–9: Deep Learning

Understanding Deep Learning
Deep Learning — Goodfellow et al.

Main goal:

Understand deep networks, backprop, optimizers, regularization, and representation learning.

Build:

MLP from scratch
CNN in PyTorch
Autograd experiments
Adam optimizer from scratch

⸻

Months 10–12: Transformers and LLMs

NLP with Transformers
Build a Large Language Model From Scratch
Speech and Language Processing selected chapters

Main goal:

Build your own small GPT.

Build:

Tokenizer
Embedding layer
Self-attention
Multi-head attention
Transformer block
Tiny GPT
Pretraining loop
Instruction fine-tuning loop

⸻

Months 13–15: Reinforcement Learning

Sutton & Barto
Algorithms for Decision Making
Deep RL Hands-On

Main goal:

Understand RL from bandits to PPO.

Build:

Multi-armed bandit
Gridworld
Q-learning
DQN
Policy gradient
Actor-critic
PPO

⸻

Months 16–18: RLHF, MoE, and Systems

Read papers:

InstructGPT
DPO
ORPO
DeepSeek-R1
Switch Transformer
Mixtral
FlashAttention
ZeRO
Megatron-LM

Build:

Reward model toy example
DPO training on preference pairs
Tiny MoE layer
Router + expert load balancing
Tiny RLHF-style PPO loop

⸻

What To Skip Initially

Do not start with these too early:

Information Geometry
Measure Theory
Advanced statistical learning theory
CUDA kernels
Distributed training
MoE papers
RLHF papers

They are valuable, but only after the foundation.

Otherwise, you will recognize words but not understand mechanisms.

⸻

The Most Important “Minimum Stack”

If you want the shortest serious path, do this:

1. Strang — Introduction to Linear Algebra
2. Bertsekas — Introduction to Probability
3. Wasserman — All of Statistics
4. Boyd — Convex Optimization
5. Prince — Understanding Deep Learning
6. Goodfellow — Deep Learning
7. Jurafsky & Martin — Speech and Language Processing
8. Raschka — Build a Large Language Model From Scratch
9. Sutton & Barto — Reinforcement Learning
10. Lapan — Deep Reinforcement Learning Hands-On
11. InstructGPT, DPO, Mixtral, DeepSeek-R1 papers

This is the best compressed route.

⸻

My Recommendation for You Specifically

Given your background as an AI developer and your interest in GraphRAG, agents, Jarvis-like systems, Highcharts agents, and LLM OS ideas, I would not study this like a traditional student.

Use this pattern:

Math concept → tiny implementation → LLM/RL connection → paper

Example:

Matrix multiplication
→ implement it
→ connect to embeddings and attention
→ read Attention Is All You Need

Another example:

KL divergence
→ implement it
→ connect to RLHF KL penalty
→ read InstructGPT

Another:

Policy gradient
→ implement REINFORCE
→ connect to PPO
→ connect to RLHF

This will make the books stick much more deeply.
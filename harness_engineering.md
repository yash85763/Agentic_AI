# Loop Engineering, Harness Engineering & Agentic Middleware Engineering
### A Complete Learning Curriculum (Table of Contents + Resources per Concept)

> **The one-line thesis:** `Agent = Model + Harness`. Holding the model fixed reveals how much engineering leverage lives in the layer *around* it. LangChain's DeepAgent jumped **13.7 points on Terminal-Bench 2.0 (52.8% → 66.5%) by changing only the harness — same model**. That gap is a large part of the discipline, though not the whole story.
>
> **How to read this document:** Concepts are ordered as a learning path. Each concept has (a) a one-paragraph "what it is," and (b) the canonical blog/paper/repo to learn it from. **Part 0** is **Loop Engineering** (the practitioner methodology — designing the *system* that prompts agents, not the prompts themselves). **Part I** is **Harness Engineering** (the agent runtime + the repo/context scaffold around a coding agent like Claude Code or Codex). **Part II** is **Agentic Middleware Engineering** (the programmable interception layer *inside* the loop — context, routing, guardrails, observability). **Part III** is the study plan. Loop Engineering is *why* you build the harness; Harness Engineering is *what* you build; Middleware Engineering is *how* the cross-cutting concerns get implemented.
>
> **Calibration note:** The result above demonstrates harness leverage while holding one model fixed; it does **not** mean models are interchangeable or that a harness can compensate for inadequate base capability. A precise working model is: `agent behavior = model + assembled context + tools + execution environment + feedback/verifiers + loop control`. The harness is where those components become a reliable system.

---

## Definitions (read these first)

**Loop engineering** = the discipline of designing the *system* that decides what prompt to write next, runs it, reads the result, and decides whether to continue — instead of hand-writing each individual prompt. The human sets a high-level goal; an orchestrator agent manages a fleet of specialized sub-agents that work in parallel, use external memory, and iterate autonomously. The scarce skill is defining what "good" and "done" mean — the *verifier*, not the model, is the bottleneck. Loop engineering is the 2026 practitioner-layer methodology that *uses* harnesses and *requires* middleware.

**Harness engineering** = everything around your AI agent except the model: the agent loop, the tool surface, memory, permissions/sandbox, hooks, observability, plus the repo-side scaffold (AGENTS.md / CLAUDE.md, docs, task specs, CI gates) that feeds and constrains it. It shapes *what the agent can see, touch, and change.*

**Agentic middleware engineering** = the programmable layer that sits *between* the core steps of the agent loop (before/after model call, before/after tool call) and implements cross-cutting concerns — summarization, context trimming, model routing, guardrails, retries, logging — without changing the model or the agent's core logic. If the harness is the body, middleware is the nervous system wiring you can edit.

The three disciplines stack: Loop Engineering is the *design methodology* (what system do I build?). Harness Engineering is the *infrastructure* (how do I build and constrain the runtime?). Middleware Engineering is the *implementation mechanics* (how do I wire cross-cutting concerns into the loop?). Learn them in that order.

---

# PART 0 — LOOP ENGINEERING

> **The origin moment.** On June 7, 2026, Peter Steinberger (creator of OpenClaw, now at OpenAI) posted twelve words that restructured how the AI engineering community talks about agents: *"You shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents."* The post reportedly crossed 6.5 million views in days and gave a name to something experienced agent builders had already started doing. Loop engineering is that name.

## 0.1 What loop engineering actually is (and where it sits)

Loop engineering is a *design methodology* layered above prompt engineering, context engineering, and harness engineering. Prompt engineering optimises a single model interaction. Loop engineering optimises the autonomous behaviour that *surrounds* it — the trigger, the state check, the decision, the execution, the feedback, and the exit condition. Your job is no longer to write the best prompt; it is to design the system that writes, runs, and evaluates prompts on its own.

The four-era lineage (useful to know):
1. **Prompt engineering (2022–2023)** — craft the words that get the model to do what you want.
2. **Context engineering (2024–2025)** — assemble the full context window optimally (Tobi Lütke, Karpathy, Anthropic).
3. **Harness engineering (2025–2026)** — design the entire runtime environment around the agent.
4. **Loop engineering (2026–present)** — design the system that orchestrates agent loops autonomously.

- **Blog (essential, the lineage):** *What Is Loop Engineering? A Complete Guide from Prompt to Harness Engineering (2026)* — Tosea.ai. https://tosea.ai/blog/loop-engineering-ai-agents-complete-guide-2026
- **Blog (practitioner definition):** *Loop Engineering: The Skill That's Replacing Prompting* — Vovance / Medium. https://medium.com/@vovance/loop-engineering-the-skill-thats-replacing-prompting-d429b000489c
- **Blog (visual explainer):** *Loop Engineering Explained Visually: From Manual Prompts to Goal-Driven AI Agents* — Techlatest / Medium. https://medium.com/@techlatest.net/loop-engineering-explained-visually-from-manual-prompts-to-goal-driven-ai-agents-f2c4d634c261

---

## 0.2 The four primitives of every loop

Every production loop is built from the same four blocks, regardless of platform:

1. **Trigger** — what starts the loop (cron/schedule, webhook/event, human message, another agent).
2. **State check** — read the current situation (tests, logs, repo, inbox, memory).
3. **Execution** — write code, call a tool, run a command, produce an artifact.
4. **Feedback / verifier** — capture the result and decide: done, iterate, or escalate.

**The central insight**: the verifier (block 4) is the bottleneck, not the model. Defining "done" in measurable, mechanical terms — passing tests, a confidence threshold, a human sign-off — is the hardest and most important design decision in any loop.

- **Blog (four primitives + four loop patterns):** *Loop Engineering (2026): Self-Prompting AI Agent Patterns* — Agent Shortlist. https://agentshortlist.com/articles/loop-engineering
- **Blog (verifier-as-bottleneck + checklist):** *Loop Engineering Guide (2026)* — AI Builder Club. https://www.aibuilderclub.com/blog/loop-engineering-guide-2026
- **Blog (state-check/decision/execution/feedback stack):** *Loop Engineering: Why the Best AI Agents in 2026 Are Built as Loops, Not Prompts* — Shaam Blog. https://shaam.blog/articles/loop-engineering-ai-agents

---

## 0.3 Open loops vs. closed loops (the most important design decision)

The open/closed distinction is the key decision rule for every loop you design. Get this wrong and you either burn money or produce unreliable output at scale.

**Open loop** — wide operational space, vague or exploratory goal, broad agent autonomy. The agent discovers its own path. Useful for research tasks, exploring unknown solution spaces, creative generation. Costs: unpredictable reasoning chains, context bloat, compounding API spend, "slop at scale" (output that looks done but misses the bar). *Use when: exploring, prototyping, research budget is available.*

**Closed loop** — human architect defines the path before execution: a clear goal, defined steps, an eval gate at each step, and an explicit stop condition. Agents still iterate — but inside your frame. Deterministic gates (test suite pass/fail, compile check) are the cheapest verifiers; non-deterministic gates (LLM reviewer, quality rubric) add power but cost. *Production default: start closed.*

- **Blog (open vs. closed framing):** *Loop Engineering Explained Visually* — Techlatest / Medium. https://medium.com/@techlatest.net/loop-engineering-explained-visually-from-manual-prompts-to-goal-driven-ai-agents-f2c4d634c261
- **Blog (budget-aware design):** *Loop Engineering Guide (2026)* — AI Builder Club. https://www.aibuilderclub.com/blog/loop-engineering-guide-2026
- **Blog (deterministic vs. non-deterministic gates):** *Loop Engineering: Why the Best AI Agents in 2026 Are Built as Loops, Not Prompts* — Shaam Blog. https://shaam.blog/articles/loop-engineering-ai-agents

---

## 0.4 The orchestrator + specialist fleet pattern

The practitioner architecture: a human sets a high-level goal → an **orchestrator agent** decomposes it and manages a **fleet of specialized sub-agents** (builder, scout, growth agent, researcher, verifier…) running in parallel, sharing external memory to track progress. The orchestrator does not execute; it routes, monitors, and synthesizes.

The graduation moment from "I ran an agent" to "I built an automated system" is **scheduling** — moving from manual trigger to a cron or event-driven automaton. This is when an orchestrated loop becomes a business process.

- **Blog (orchestrator fleet + scheduling graduation):** *Loop Engineering: The Guide for AI Agents* — Lushbinary. https://lushbinary.com/blog/loop-engineering-ai-coding-agents-guide/
- **Blog (four loop patterns including heartbeat/cron):** *Loop Engineering (2026): Self-Prompting AI Agent Patterns* — Agent Shortlist. https://agentshortlist.com/articles/loop-engineering
- **Primary (Anthropic):** *Building Effective Agents* — orchestrator-workers and evaluator-optimizer patterns (the infrastructure behind the methodology). https://www.anthropic.com/research/building-effective-agents

---

## 0.5 The academic foundations of loop patterns (the papers)

Loop engineering is a practitioner name for a convergence of research threads that have been building for several years. These papers are the theoretical foundation; the 2026 practitioner movement is their production crystallisation.

### The ReAct loop (Reason + Act) — the atomic loop
The foundational paper for all modern agent loops. The model alternates between generating a verbal *thought* (reasoning) and taking an *action* (tool call), then observing the result — cycling until done. Loops are ReAct at their core.
- **Paper (essential):** *ReAct: Synergizing Reasoning and Acting in Language Models* — Yao et al., ICLR 2023. The thought-action-observation paradigm. https://arxiv.org/abs/2210.03629

### Self-Refine — the self-improvement loop
Without any additional training data or RL, a single LLM can generate output, provide feedback on it, and refine it — looping through FEEDBACK → REFINE → FEEDBACK until quality is sufficient. Improved outputs by ~20% across 7 tasks. This is the "generate → critique → revise" closed loop.
- **Paper (essential):** *Self-Refine: Iterative Refinement with Self-Feedback* — Madaan et al., NeurIPS 2023. https://arxiv.org/abs/2303.17651

### Reflexion — verbal reinforcement learning (the memory loop)
Rather than updating model weights, Reflexion agents verbally reflect on failure signals and store their reflections in an episodic memory buffer — improving decisions in *subsequent* trials. This is the loop that learns from its own mistakes across runs without fine-tuning.
- **Paper (essential):** *Reflexion: Language Agents with Verbal Reinforcement Learning* — Shinn, Cassano, Gopinath, Narasimhan, Yao; NeurIPS 2023. https://arxiv.org/abs/2303.11366
- **Repo:** https://github.com/noahshinn/reflexion

### ReST + ReAct — iterative fine-tuning from loop trajectories
Combines a ReAct-style agent with ReST (growing-batch RL on its own reasoning traces) for iterative self-improvement and distillation. Two iterations of the loop produce a small model comparable to a 100× larger prompted model.
- **Paper:** *ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent* — Aksitov et al., 2023. https://arxiv.org/abs/2312.10003

### Agentic Self-Learning (ASL) — the fully closed self-improving loop
A fully closed-loop, multi-role RL framework: a Prompt Generator creates tasks, a Policy Model executes them, a Generative Reward Model evaluates them — all in a shared environment. No external supervision. This is the research frontier of loops that improve their own prompts.
- **Paper (research frontier):** *Towards Agentic Self-Learning LLMs in Search Environment* (ASL framework). https://arxiv.org/pdf/2510.14253

### AutoRefine — reusable expertise extracted from loop trajectories
Extracts successful trajectory patterns as reusable skills and guidelines. Procedural subtasks become specialist subagents; static knowledge becomes skill patterns. A maintenance mechanism scores, prunes, and merges patterns to prevent degradation. This is the loop that *builds its own harness over time*.
- **Paper:** *AutoRefine: From Trajectories to Reusable Expertise for Continual LLM Agent Refinement* (2026). https://arxiv.org/html/2601.22758v1

### Self-play SWE-RL — loops that generate their own training signal
An agent trains itself by injecting and repairing software bugs of increasing complexity, with no human-labeled issues. RL in a self-play setting. +10.4 points on SWE-bench Verified. The loop *is* the training run.
- **Paper (research frontier):** *Toward Training Superintelligent Software Agents through Self-Play SWE-RL*. https://arxiv.org/pdf/2512.18552

### Verified multi-agent orchestration
Formalizes the plan-execute-verify pattern for multi-agent systems with specialized agents, each with clear contracts. Brings together ReAct-style execution, tool use, and output verification into a coherent production architecture.
- **Paper:** *Verified Multi-Agent Orchestration: A Plan-Execute-Verify Framework*. https://arxiv.org/pdf/2603.11445

---

## 0.6 Failure modes of loops (controlled autonomy, not unlimited autonomy)

Loop engineering fails loudly — and expensively — if failure modes are not engineered as carefully as the success path. The design goal is *controlled autonomy*, not unlimited autonomy. Key failure modes to engineer against:

- **Runaway open loops** — no stop condition, reasoning chains compound, API spend explodes.
- **Slop at scale** — output that looks finished but misses the actual bar (usually a weak verifier).
- **Context pollution across iterations** — stale information from prior loop passes corrupts the current one; requires memory management (§4, §8).
- **Stale harness assumptions** — the loop encodes "what the model can't do yet" but those assumptions don't get revisited as models improve (§4.3).
- **Human oversight erosion** — scheduling removes the human from the loop; maintain explicit escalation paths and checkpoints for destructive or irreversible actions (§5, §10.2).

- **Blog (failure modes + controlled-autonomy framing):** *Loop Engineering: The Skill That's Replacing Prompting* — Vovance / Medium. https://medium.com/@vovance/loop-engineering-the-skill-thats-replacing-prompting-d429b000489c
- **Cross-reference:** §4.3 (harness assumptions decay), §10.2 (human-in-the-loop checkpoints).

---

# PART I — HARNESS ENGINEERING


## 1. Foundations & Mental Model

> **Learning goal.** By the end of this chapter, you should be able to look at any “agent” product and separate six things that are often blurred together: the model, the current context, the loop controller, the tool surface, the execution environment, and the verifier. You should also be able to draw the core loop from memory and explain why a model that writes good code can still produce an unreliable coding agent.

### 1.1 Why the harness is the job
The slogan `Agent = Model + Harness` is directionally right, but it is a compression. A more precise practical equation is:

```text
Agent behavior
  = model capability
  + assembled context
  + available tools and their interfaces
  + execution environment and permissions
  + feedback and verification
  + control logic for retry, stop, escalation, and memory
```

A model is a conditional generator: given the tokens and tool schemas in its current context, it proposes the next tokens or a tool call. It does **not** inherently know which repository it is in, which files matter, whether a shell command is safe, whether its code compiles, whether it should retry, or whether its answer has met the real requirement. Those are system responsibilities.

The harness is the surrounding runtime that turns a model call into a controlled sequence of **observe → decide → act → receive feedback → continue, stop, or escalate**. This is why two agents using the same model can have dramatically different reliability.

| Same base model, different system | Weak harness | Strong harness |
|---|---|---|
| Initial context | “Fix the authentication bug.” | Issue, scoped repo instructions, relevant test commands, environment state, and tool descriptions. |
| Action surface | Text response only or vague tools. | Search, read, edit, shell, tests, version control, and domain tools with clear contracts. |
| Safety | Prompt asks the model not to do dangerous things. | Sandboxed execution, least-privilege credentials, policy gates, and approval checkpoints. |
| Feedback | Model self-declares completion. | Unit tests, integration tests, linters, security checks, schema checks, or human review. |
| Failure recovery | One incorrect answer ends the run. | Retry policies, loop detection, structured errors, rollback, and escalation. |
| Learning from runs | Conversation disappears. | Durable traces, plans, checkpoints, memory, and evaluated skill updates. |

The model supplies general capability. The harness converts that capability into bounded, inspectable, task-specific behavior. The harness is therefore not “extra plumbing.” It determines what world the model sees, what actions it can attempt, what consequences it can cause, and what evidence can prove it succeeded.

#### Do not overcorrect: the model still matters
Harness engineering is not a claim that models are commodities. A harness cannot reliably rescue a model that cannot follow instructions, select tools, interpret tool output, write valid code, recover after failure, or use relevant instructions. What harness work does is prevent capable models from wasting ability through missing context, poorly designed tools, unverified outputs, permissive environments, or unbounded loops.

The DeepAgents Terminal-Bench result in this curriculum is a useful example. Holding a model fixed, changes to environment context, verification behavior, loop detection, and reasoning-budget policy materially changed task success. That demonstrates that the **system around the model** can be a high-leverage variable. It does not demonstrate that the model is unimportant.

#### Six neighboring disciplines, separated cleanly

| Discipline | Primary question | Typical artifact | What it cannot guarantee by itself |
|---|---|---|---|
| **Prompt engineering** | What should I tell the model to do? | System prompt, task instruction, few-shot examples. | That the instruction will be followed or safely enforced. |
| **Context engineering** | What information should be present in the finite context window now? | Context builder, retrieval policy, compaction, summaries, memory selection. | That the model can act on the information or that actions are safe. |
| **Harness engineering** | How do I execute and govern the model-tool-feedback loop? | Loop controller, tool runtime, state, policies, validators, traces. | That the underlying model has enough capability for the task. |
| **Environment engineering** | What can the running agent actually touch? | Container/VM, filesystem mounts, network rules, IAM, secrets policy. | That the model chooses the correct action within the allowed boundary. |
| **Middleware engineering** | Where do cross-cutting policies alter the loop? | Before/after model and tool hooks, routing, trimming, guardrails, logging. | That the whole agent architecture is well designed. |
| **Loop engineering** | What trigger, state check, verifier, and exit rule make this a useful autonomous process? | Orchestration plan, stop conditions, eval gates, escalation policy. | That every underlying component is implemented correctly. |

A helpful enforcement ladder is:

```text
Instruction in AGENTS.md      = advice the model may ignore
Pre-tool policy / hook        = harness enforcement before execution
Read-only mount / no network  = environment enforcement the model cannot bypass
```

This is why a prompt that says “do not touch production” is never the final safety mechanism. A reliable design makes the dangerous resource unavailable or requires an approval boundary before the action runs.

#### What to read for this section
- **Primary (OpenAI):** *Harness engineering: leveraging Codex in an agent-first world.* This is the best practitioner account of how the human engineering role shifts from writing every line to increasing agent legibility, specifying intent, and building feedback loops. https://openai.com/index/harness-engineering/
- **Primary (Anthropic):** *Building Effective Agents.* Use it for the workflow-versus-agent distinction and the augmented-LLM mental model. https://www.anthropic.com/research/building-effective-agents
- **Primary (Anthropic):** *Effective context engineering for AI agents.* Use it for the definition of context as the finite set of tokens present at inference and the shift from prompt wording to managing the entire context state. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- **Survey (new):** *Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems.* A useful 2026 survey that treats code as the operational substrate for agent reasoning, action, environment modeling, and execution-based verification. https://arxiv.org/abs/2605.18747
- **Survey (new):** *Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering.* A systems-level frame: memory externalizes state across time, skills externalize procedure, protocols externalize coordination structure, and the harness coordinates them into governed execution. https://arxiv.org/abs/2604.08224

### 1.2 The agentic loop (the irreducible core)
At its core, an agent is a repeated state transition:

```text
state at time t
  → assemble context
  → model chooses a response or tool action
  → execute action under policy and sandbox
  → receive an observation
  → update state
  → repeat or terminate
```

In compact form:

```text
while not terminal:
    context = assemble_context(state)
    response = model(context, available_tools)

    if response is a final answer:
        verdict = verify(response, state)
        if verdict.passed:
            return response
        state.add(verdict.feedback)
        continue

    for tool_call in response.tool_calls:
        decision = policy_gate(tool_call, state)
        if decision.requires_approval:
            return escalate(tool_call)
        if decision.denied:
            state.add(decision.reason)
            continue

        observation = sandbox.execute(tool_call)
        trace.record(tool_call, observation)
        state.add(observation)

    if budget_exhausted(state) or loop_detected(state):
        return stop_or_escalate(state)
```

This pseudo-code contains nearly every major topic in the rest of the curriculum:

- `assemble_context` leads to repo instructions, skills, retrieval, memory, compaction, and context middleware.
- `available_tools` leads to tool design, MCP, tool schemas, and programmatic tool calling.
- `policy_gate` and `sandbox.execute` lead to permissions, hooks, approvals, sandboxing, and containment.
- `verify` leads to tests, CI, graders, human checkpoints, and eval-driven improvement.
- `trace.record` leads to observability, failure diagnosis, and self-evolving harnesses.
- `budget_exhausted` and `loop_detected` lead to controlled autonomy rather than runaway iteration.

#### Turn, loop, task, and session are different units
These terms are easy to blur:

- A **model call** is one inference request.
- A **tool step** is one attempted action plus its observation.
- A **loop iteration** is a model decision followed by zero or more tool steps.
- A **task** is the user goal, such as “fix this bug.”
- A **session** is the persistent record of task state, messages, observations, plans, artifacts, and checkpoints that may span multiple context windows.

A coding agent may take dozens of model calls and hundreds of tool steps to complete one task. The main output may not be the final chat message. It may be a changed file, a commit, an updated deployment manifest, or a passing test suite.

#### Codex is a concrete reference implementation
OpenAI’s *Unrolling the Codex agent loop* makes the abstract loop tangible. Before the first inference call, Codex constructs an input from model instructions, tool definitions, sandbox and approval information, repository instructions such as `AGENTS.md`, configured skill metadata, local environment information, and the user’s message. When the model emits a tool call, Codex executes it, appends the observation into the next request, and samples the model again. The sequence continues until the model returns an assistant message instead of another tool call.

The same article exposes two engineering constraints that explain why harness design is substantive:

1. **Context growth:** tool results, plans, errors, and history can consume the finite context window. The harness needs compaction and memory strategies.
2. **Inference cost and latency:** repeated calls are expensive. Stable prompt prefixes, tool ordering, cache behavior, and configuration changes can materially affect performance.

That is why the loop is not just `while(tool_calls)`. It is a resource-constrained control system.

#### Workflows versus agents
Anthropic makes a useful architectural distinction:

- **Workflows** use LLMs and tools through predefined code paths. The system determines the sequence.
- **Agents** let the model dynamically direct its own process and tool usage. The model determines much of the sequence.

Neither is universally better. Use workflows when the path is known, precision matters, and the cost of wandering is high. Use agents when the path cannot be predicted in advance and the system needs flexible tool selection or exploration. A closed loop can include agentic substeps while still having deterministic gates around them.

#### What to read for this section
- **Primary (OpenAI, essential):** *Unrolling the Codex agent loop.* Read it slowly. It shows prompt construction, tool execution, message roles, caching constraints, and compaction as concrete runtime mechanisms. https://openai.com/index/unrolling-the-codex-agent-loop/
- **Primary (Anthropic):** *Building Effective Agents.* Focus on the distinction between workflows and agents, plus the orchestrator-workers and evaluator-optimizer patterns. https://www.anthropic.com/research/building-effective-agents
- **Foundational paper:** *ReAct: Synergizing Reasoning and Acting in Language Models.* The thought/action/observation pattern is the academic ancestor of this loop. https://arxiv.org/abs/2210.03629
- **Source code:** `openai/codex`. Read the code only after the OpenAI article, so the source has a mental map around it. https://github.com/openai/codex

### 1.3 The architecture of a real harness
A production harness is easiest to reason about as three interacting planes.

```text
                         CONTROL PLANE
         session state · context assembly · policies · budgets
       retries · stopping rules · memory · approvals · middleware
                                  │
                                  ▼
                          MODEL PLANE
          inference · tool selection · planning · response generation
                                  │ tool calls
                                  ▼
                    EXECUTION + FEEDBACK PLANE
     tools · MCP · shell · filesystem · APIs · sandbox · tests · graders
     traces · logs · checkpoints · artifacts · human review decisions
                                  │ observations
                                  └─────────────── back to CONTROL PLANE
```

The model plane is important, but it is only one plane. Most production reliability work occurs in the control and execution-feedback planes.

| Component | Core responsibility | Failure when poorly designed |
|---|---|---|
| **Context assembler** | Chooses which instructions, files, memories, tool schemas, and observations enter the next call. | The agent never sees the requirement, sees too much irrelevant material, or follows stale instructions. |
| **Tool interface** | Defines actions with names, arguments, examples, output formats, errors, and side effects. | The model selects the wrong tool or produces malformed arguments. |
| **Tool executor** | Actually performs the action and returns a usable observation. | Tool output is missing, ambiguous, unstructured, or detached from the relevant state. |
| **Policy / permission gate** | Allows, denies, redirects, or requests approval for actions. | The agent can perform destructive, costly, or prohibited actions. |
| **Environment boundary** | Enforces filesystem, network, process, credential, and runtime isolation. | A hallucination or prompt injection has a large blast radius. |
| **Verifier** | Tests whether the system has achieved the real success condition. | The agent says “done” while the code fails or the task requirement is unmet. |
| **Trace and observability layer** | Preserves the evidence needed to inspect and improve behavior. | A harness bug, tool outage, model failure, and sandbox failure all look the same. |

#### The verifier is the trust boundary
A model’s declaration that it is finished is only a candidate completion. The harness should decide whether that completion is accepted.

For a coding task, a verifier could include:

```text
required tests pass
AND formatter/linter passes
AND changed files stay within scope
AND no prohibited dependency or secret is introduced
AND task-specific acceptance checks pass
```

For a company-profiling agent, a verifier could require:

```text
every required profile field is present
AND every non-null claim has an evidence pointer
AND document-derived facts are separated from web-derived facts
AND unsupported values are explicitly marked unknown
AND low-confidence inferences enter a human-review queue
```

The point is not that every task needs an enormous eval suite. The point is that “done” should be made as mechanical as the task allows. Deterministic verification is generally cheaper and more reliable than asking another model whether the first model did a good job.

#### Claude Code and the “no RAG” misconception
The useful claim is not that Claude Code does no retrieval. It does. The more accurate claim is:

> Coding agents need not rely on a traditional, precomputed vector-RAG index to discover repository context. They can use agent-directed filesystem retrieval such as globbing, grep, symbol search, file reading, and shell commands, then choose the next retrieval action based on the results.

Compare the two patterns:

```text
Traditional repository RAG
repo → chunk → embed → index → retrieve top-k chunks → model

Agent-directed filesystem retrieval
model sees task → searches paths/symbols → opens evidence → updates hypothesis
               → searches again only where needed → edits/tests
```

Pre-indexed retrieval can be fast and useful, particularly for large or static collections. Agent-directed retrieval can be adaptive and fresh because the agent explores the live repository. In practice, robust systems often use a hybrid: stable instructions and high-value architecture context up front, then just-in-time search for task-specific evidence.

Do not treat precise tool counts, undocumented compaction thresholds, or internal Claude Code implementation details from third-party guides as permanent facts. Harness internals change quickly. Prefer first-party documentation for behavior guarantees and use source analyses as learning aids.

#### New research that sharpens the Chapter 1 mental model
- **Code as Agent Harness (2026):** proposes a three-layer view: harness interface, harness mechanisms (planning, memory, tool use, feedback control), and harness scaling to multi-agent systems. Its open problems are exactly the right next questions: evaluation beyond final success, incomplete feedback, regression-free improvement, shared state, human oversight, and multimodal environments. https://arxiv.org/abs/2605.18747
- **Harnesses for Inference-Time Alignment over Execution Trajectories (2026):** argues that more decomposition and guidance are not automatically better. It separates harnesses into task decomposition and guided execution, then highlights over-decomposition, over-pruning, and hallucinated execution as real failure modes. https://arxiv.org/abs/2605.21516
- **A Benchmark for Context Retrieval in Coding Agents / ContextBench (2026):** measures not only final task completion but process-level context recall, precision, and efficiency. This motivates a critical distinction: *context retrieved* is not necessarily *context used correctly*. https://arxiv.org/abs/2602.05892

#### Chapter 1 mastery check
Before moving to the repo-side scaffold, make sure you can answer these without notes:

1. Why is “model says it is finished” not a sufficient stop condition?
2. What is the difference between a tool schema, a tool executor, and a sandbox?
3. Which parts of a coding agent determine what it can **see**, what it can **do**, and what it can **change**?
4. Why is a policy written in `AGENTS.md` weaker than a pre-tool approval hook, and why is a hook weaker than an environment-level capability boundary?
5. Why can a fixed model perform better under one harness than another without implying that the model is unimportant?
6. Why is adaptive repository navigation still retrieval, even when no vector database is present?

**Deliverable:** Draw the three-plane architecture for one agent you want to build. For every box, name one likely failure mode and one observable signal that would reveal it.


---

## 2. The Repo-Side Scaffold (Context Engineering for Coding Agents)

### 2.1 AGENTS.md / CLAUDE.md — the instruction file
The near-universal standard (read by Claude Code, Codex, Cursor, Aider, Copilot, Gemini CLI, etc.). It tells the agent how to navigate the repo, run tests, and follow conventions. Treat it as code: version-controlled, reviewed, owned.
- **Standard guide:** *AGENTS.md Complete Guide for Engineering Teams (2026)* — structure, monorepo patterns, anti-patterns. https://blog.buildbetter.ai/agents-md-complete-guide-for-engineering-teams-in-2026/
- **Primary (OpenAI):** *Custom instructions with AGENTS.md* — Codex docs. https://developers.openai.com/codex/guides/agents-md

### 2.2 AGENTS.md as table-of-contents, not encyclopedia (the scaling lesson)
**The single most important repo-scaffold lesson.** The "one big AGENTS.md" fails: context is scarce, too much guidance becomes non-guidance, it rots, and it's unverifiable. The fix: a ~100-line AGENTS.md that acts as a *map* pointing into a structured `docs/` system-of-record. This is from the team that shipped ~1M LOC / 1,500 PRs in 5 months with agents.
- **Primary (OpenAI, essential):** *Harness engineering: leveraging Codex in an agent-first world*. https://openai.com/index/harness-engineering/

### 2.3 Context engineering (the umbrella discipline)
Treat context as a finite, precious resource. Just-in-time retrieval (glob/grep/bash over the filesystem) beats pre-indexing for dynamic content; hybrid strategies load some context up front. Sub-agents isolate context. This is the theory behind §2.1–2.2.
- **Primary (Anthropic):** *Effective context engineering for AI agents*. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### 2.4 Agent Skills (composable, loadable expertise)
A `SKILL.md` folder packages procedural knowledge + scripts that the agent discovers and loads *only when relevant* — keeping context lean. Now an open cross-platform standard; both Codex and Claude support skills.
- **Primary (Anthropic):** *Equipping agents for the real world with Agent Skills*. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- **Primary (OpenAI):** Codex Changelog — Skills layout (`.codex/skills`, `$skill-creator`, `$skill-installer`). https://developers.openai.com/codex/changelog

---

## 3. Tools (the Agent-Computer Interface)

### 3.1 Designing tools the model can actually use
Tools are the agent's primary actions and dominate the context window, so design them with as much care as a human UI ("ACI" = agent-computer interface). Clear names, examples, edge cases, minimal overhead.
- **Primary (Anthropic):** *Writing effective tools for AI agents — using AI agents* (includes the eval-driven refinement loop and the tool-eval cookbook). https://www.anthropic.com/engineering/writing-tools-for-agents
- **Primary (Anthropic):** *Building Effective Agents*, Appendix 2 "Prompt Engineering your Tools." https://www.anthropic.com/research/building-effective-agents

### 3.2 Programmatic / code-mode tool calling
Instead of one round-trip per tool call, the model writes code that orchestrates many tool calls internally; only final stdout enters context. Major token/latency win for tool-heavy loops.
- **Reference:** claude-code-ultimate-guide architecture.md (Programmatic Tool Calling section, GA Feb 2026). https://github.com/FlorianBruniaux/claude-code-ultimate-guide/blob/main/guide/core/architecture.md
- **Primary docs:** Anthropic Advanced Tool Use / Programmatic Tool Calling — https://docs.claude.com (search "programmatic tool calling").

### 3.3 MCP (Model Context Protocol) — the extension surface
The portable standard for connecting agents to external tools/data. The primary portability layer across harnesses; both Codex and Claude Code consume MCP servers.
- **Primary:** Model Context Protocol spec & docs. https://modelcontextprotocol.io

---

## 4. Long-Running Agents & Memory (crossing context windows)

### 4.1 The shift-change problem and the initializer/coder pattern
Long tasks span many context windows; each new session starts blind. Solution: an **initializer agent** (writes `init.sh`, `claude-progress.txt`, initial git commit) + a **coding agent** that makes incremental progress and leaves structured artifacts. Inspired by how human engineers actually work.
- **Primary (Anthropic, essential):** *Effective harnesses for long-running agents* (+ quickstart code). https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

### 4.2 Compaction, memory tools, and persistent state
When context fills, summarize-and-replace (compaction). Beyond that: file-based memory tools, plans as first-class checked-in artifacts, and Codex's two-phase memory pipeline (extract → consolidate, capped & injected at session start).
- **Primary (Anthropic):** memory tool + context management — see *Effective context engineering* (§2.3) and the memory/context-management cookbook referenced there.
- **Source analysis:** Zylos Codex architecture (memory_summary.md, 5k-token cap, SQLite job leases). https://zylos.ai/research/2026-03-26-openai-codex-cli-architecture-multi-runtime-patterns/
- **Primary (OpenAI):** Codex execution-plan / docs-as-system-of-record pattern — *Harness engineering* (§2.2 link).

### 4.3 Harnesses encode (decaying) assumptions
Crucial maturity lesson: harness logic encodes "what the model can't do yet" — and goes stale as models improve (e.g., context-anxiety resets that became dead weight on Opus 4.5). Design harnesses to be questioned and stripped back over time.
- **Primary (Anthropic):** *Scaling Managed Agents: Decoupling the brain from the hands*. https://www.anthropic.com/engineering/managed-agents

---

## 5. Environment, Sandboxing & Safety (the blast radius)

### 5.1 Sandboxing and approval models
Bound what the agent can touch: OS user, filesystem scope, network egress, approval gates. Codex defaults to bubblewrap sandbox + an execpolicy rule DSL with hardcoded-banned vectors (shell interpreters, `sudo`); Codex Cloud runs with internet disabled during execution.
- **Primary (OpenAI):** *Introducing Codex* (container isolation, no-internet execution, refusal training). https://openai.com/index/introducing-codex/
- **Source analysis:** Zylos (execpolicy DSL, bwrap hardening). https://zylos.ai/research/2026-03-26-openai-codex-cli-architecture-multi-runtime-patterns/
- **Primary (OpenAI):** *Unrolling the Codex agent loop* (the developer-role sandbox message; note: MCP tools are NOT sandboxed by Codex — they self-enforce). https://openai.com/index/unrolling-the-codex-agent-loop/

### 5.2 Containment as capabilities scale
As agents grow more capable, the engineering question becomes how to cap blast radius. Anthropic's containment learnings across claude.ai, Claude Code, and Cowork.
- **Primary (Anthropic):** Engineering blog hub (latest containment post). https://www.anthropic.com/engineering

### 5.3 Hooks (deterministic enforcement points)
Hooks (e.g., `user_prompt_submit`, pre/post tool) let you enforce policy the model *cannot* bypass — the difference between "advice the model may ignore" and real enforcement. This is layer 4 of the 5-layer model.
- **Reading path:** DEV 5-layer guide (Memory → Tools → Permissions → Hooks → Observability). https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j
- **Primary docs:** Claude Code hooks + Codex hooks (user_prompt_submit) — https://docs.claude.com and https://developers.openai.com/codex/changelog

---

## 6. The Production Harness (validation, CI gates, throughput)

### 6.1 Validation, test harnesses, and CI gates for AI-generated code
The "missing layer" that turns vibe coding into shipping: output validators, test harnesses for AI changes, CI gates, security-scan gates, PR-review automation. Codex Autofix-in-CI and Codex PR review are the productized versions.
- **Book (practical build-alongs):** *Harness Engineering for AI Coding Agents* — Leandro Calado (AGENTS.md generator, context-pack builder, validator, CI gate, security gate, PR-review harness). https://www.amazon.com/Harness-Engineering-Coding-Agents-Guardrails/dp/B0H1MVTHXM
- **Primary (OpenAI):** Codex product page (PR review, Automations, CI/CD, worktrees). https://openai.com/codex/
- **Case study:** *Harness engineering* (3→7 engineers, 3.5 PRs/eng/day, docs-as-system-of-record, quality grading per domain). https://openai.com/index/harness-engineering/

### 6.2 Meta-harnesses (orchestrating multiple agents under one roof)
A meta-harness sits *above* individual coding agents (Claude Code, Codex, custom) with shared sessions and policies — useful for portability across model bans/outages.
- **Blog:** *What Is an Agent Harness? The Architecture Behind Claude Code, Codex, and Cursor* — MindStudio (covers OmniAgent-style meta-harnesses + the 6 harness components). https://www.mindstudio.ai/blog/what-is-agent-harness-architecture-explained

### 6.3 The Agent SDK (build your own harness)
Anthropic renamed the Claude Code SDK → **Claude Agent SDK** because the same harness powers research, note-taking, etc. — not just coding. This is the productized, reusable harness. Includes subagents, agentic vs. semantic search guidance, folder-as-context.
- **Primary (Anthropic, essential):** *Building agents with the Claude Agent SDK*. https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk

---

# PART II — AGENTIC MIDDLEWARE ENGINEERING

> Middleware is *how* the loop's cross-cutting concerns get implemented. Where Part I gave you the mental model, Part II gives you the programmable interception points.

## 7. The Middleware Concept

### 7.1 What middleware is and the lifecycle hooks
Middleware intercepts data flow *between* core agent steps. Two fundamental powers: **update context** (modify/persist state) and **jump in the lifecycle** (skip a step, repeat a model call, short-circuit a tool). This is the mechanism that makes context engineering practical in code.
- **Primary (LangChain, essential):** *Context engineering in agents* — the canonical middleware model (Model Context / Tool Context / life-cycle hooks). https://docs.langchain.com/oss/python/langchain/context-engineering
- **Repo:** LangChain / LangGraph (read the middleware + interrupt/checkpoint APIs). https://github.com/langchain-ai/langchain

### 7.2 Design patterns: the control structures around the model
Reactive loops (ReAct), hierarchical supervision, orchestrator-workers, evaluator-optimizer, graph-based memory. A successful agentic system is defined by the robustness of these control structures, not just model IQ.
- **Primary (Anthropic):** *Building Effective Agents* (the pattern catalog). https://www.anthropic.com/research/building-effective-agents
- **Blog:** *Design Patterns for Agentic AI and Multi-Agent Systems* — AppsTek (token-monitoring middleware, hierarchical supervision, graph memory). https://appstekcorp.com/blog/design-patterns-for-agentic-ai-and-multi-agent-systems/
- **Blog:** *Agentic Architecture: Designing AI Agents for Enterprise* — Algolia (orchestrator vs. peer-to-peer vs. hybrid coordination; agent boundaries/contracts). https://www.algolia.com/blog/ai/agentic-architecture

---

## 8. Context Middleware (the highest-leverage layer)

### 8.1 Summarization, trimming, and offloading as middleware
The most common lifecycle pattern: auto-condense conversation history when it grows. Distinguish transient trimming (this call only) from persistent summarization (saved to state). Token-monitoring middleware triggers at a % of the limit.
- **Primary (LangChain):** *Context engineering in agents* (summarization vs. trimming, persistent vs. transient). https://docs.langchain.com/oss/python/langchain/context-engineering
- **Primary (Anthropic):** *Effective context engineering* (compaction, sub-agent isolation, JIT retrieval). https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### 8.2 Recoverable context & decoupling storage from management
Irreversible compaction is dangerous — you can't know which tokens future turns need. Anthropic's answer: make the *session* a durable, interrogable object, and push arbitrary, model-specific context management into the harness/middleware layer (transform fetched events before they hit context; optimize for prompt-cache hit rate).
- **Primary (Anthropic, essential):** *Scaling Managed Agents: Decoupling the brain from the hands*. https://www.anthropic.com/engineering/managed-agents

### 8.3 Adaptive / parallel context-management routing (research frontier)
Static context strategies can't adapt as accumulated context's usefulness shifts mid-task. State-aware frameworks expand multiple context-managed branches in parallel and use lookahead routing to pick the best continuation — matching static methods with up to 3× fewer turns.
- **Paper:** *AgentSwing: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents*. https://arxiv.org/pdf/2603.27490
- **Paper (terminal-agent scaffolding/context survey):** *Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering*. https://arxiv.org/pdf/2603.05344

### 8.4 Active context management is becoming an agent action (research update)
The next step beyond automatic compaction is to treat working-memory management as an explicit, evaluable part of the loop. The agent or its middleware decides whether to preserve, compress, skip, roll back, excerpt, or delete context. This changes the question from “When should we summarize?” to “What state should the agent carry forward, at what fidelity, and why?”

This research cluster adds four important distinctions:

1. **Retrieved context is not the same as used context.** A coding agent can search the correct file yet ignore the critical information when it edits. Context quality therefore needs process-level evaluation, not only final task success.
2. **Persistent state should be externalized.** Long-horizon agents need a durable workspace or state abstraction, rather than a giant conversation transcript that grows forever.
3. **Context-management strategies themselves can become skills.** The policy that decides what to keep, summarize, or discard may be learned and improved from trajectory evidence.
4. **Context operations have failure modes.** Compressing too aggressively loses evidence; keeping everything causes distraction and cost; rollback can revive useful branches but can also reintroduce stale assumptions.

- **Paper:** *A Benchmark for Context Retrieval in Coding Agents (ContextBench).* Tracks retrieval recall, precision, and efficiency over coding-agent trajectories. Use it to learn why “the agent saw the file” does not prove it used the file correctly. https://arxiv.org/abs/2602.05892
- **Paper:** *Meta Context Engineering via Agentic Skill Evolution.* Co-evolves context-engineering skills with the context files/code those skills manipulate. https://arxiv.org/abs/2601.21557
- **Paper:** *InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents.* Externalizes durable state into a file-centric workspace and rebuilds a bounded reasoning context from a state snapshot plus recent actions. https://arxiv.org/abs/2601.03204
- **Paper:** *LongSeeker: Elastic Context Orchestration for Long-Horizon Search Agents.* Introduces Context-ReAct operations such as Skip, Compress, Rollback, Snippet, and Delete. https://arxiv.org/abs/2605.05191

---

## 9. Routing & Model-Selection Middleware

### 9.1 Model routing
Route easy/common requests to small fast models (Haiku) and hard ones to flagship models (Opus/Sonnet); tune reasoning/thinking budget per task. A core middleware decision made per-call.
- **Primary (Anthropic):** *Building Effective Agents* (routing workflow + reasoning-budget tuning). https://www.anthropic.com/research/building-effective-agents

### 9.2 LLM gateways / agent gateways (the infra form of routing)
Provider-agnostic middleware: unified API across OpenAI/Anthropic/etc., caching, token tracking, cost attribution, plus enforcement (rate limits, content filters, access). Where routing meets governance.
- **Blog:** *Top Agent Gateways* — TrueFoundry. https://www.truefoundry.com/blog/top-agent-gateways

---

## 10. Guardrail Middleware (input/output validation & safety)

### 10.1 The guardrail evolution: rule-based → model-based → reasoning
The arc: predefined-lexicon filters (transparent, inflexible) → fine-tuned classifiers (LLaMA Guard, NeMo Guardrails, Aegis) → reasoning/lifelong-learning guardrails that assess actions reflectively. Guardrails are semantic checks on what an agent may operate on, applied as input/output middleware.
- **Survey/paper (essential):** *Agentic Web: Weaving the Next Web with AI Agents* (§ on guardrails). https://arxiv.org/pdf/2507.21206
- **Blog:** *Context engineering: LLM evolution for agentic AI* — Elastic (guardrails, A2A, ReasoningBank, RAG-MCP, state). https://www.elastic.co/search-labs/blog/context-engineering-llm-evolution-agentic-ai

### 10.2 Human-in-the-loop checkpoints & approvals
Pause for human review before irreversible actions (financial txns, deletes). Zero-Trust access (agent touches only what it needs), prompt-injection defense, deterministic tools for precision work. Implemented as interrupt/checkpoint middleware.
- **Primary (Anthropic):** *Building Effective Agents* (guardrails & human checkpoints). https://www.anthropic.com/research/building-effective-agents
- **Repo:** LangGraph interrupts/checkpointers (the mechanism). https://github.com/langchain-ai/langchain

---

## 11. Observability & Evaluation Middleware

### 11.1 Observability (layer 5 — you can't improve what you can't see)
Log every model call, tool call, token count, latency, and error as middleware. Without it, a harness bug, a dropped event, and a dead container all look identical.
- **Reading path:** DEV 5-layer guide (Observability layer). https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j
- **Primary (Anthropic):** *Managed Agents* (why opaque event streams make debugging impossible). https://www.anthropic.com/engineering/managed-agents

### 11.2 Eval-driven harness/tool improvement
Generate realistic eval tasks, measure tool-use accuracy + token/latency/error metrics, let agents critique their own tool definitions. Small description tweaks → large gains (Anthropic hit SOTA on SWE-bench Verified this way). The "measure → improve → ship" loop.
- **Primary (Anthropic):** *Writing effective tools for AI agents* (+ tool-evaluation cookbook). https://www.anthropic.com/engineering/writing-tools-for-agents
- **Primary (OpenAI):** *OpenAI for Developers in 2025* (evals, graders, tuning maturing into a repeatable loop). https://developers.openai.com/blog/openai-for-developers-2025

### 11.3 Evaluating a self-evolving harness: improvement is a release, not one score (research update)
A self-modifying harness should be treated like a production release candidate, not like a prompt experiment. A higher score on the latest batch can hide overfitting, regression on earlier tasks, a cost explosion, or an intermediate harness version that is better than the final one.

A serious evaluation bundle should include:

```text
training tasks used to propose an update
→ frozen update-validation tasks
→ held-out in-domain tasks
→ held-out transfer / out-of-domain tasks
→ replay tasks for regression detection
→ token, latency, tool-error, and cost records
→ saved harness snapshot and rollback path
```

- **Paper:** *SEAGym: An Evaluation Environment for Self-Evolving LLM Agents.* Builds this release-like evaluation perspective around train, validation, test, replay, transfer, cost, and snapshot records. https://arxiv.org/abs/2606.17546
- **Paper:** *Harness Updating Is Not Harness Benefit.* Separates the ability to create a harness update from the ability of a task-solving agent to benefit from that update later. This is a crucial warning: a skill or prompt may look good in isolation yet fail to be retrieved, followed, or useful for the receiving model. https://arxiv.org/abs/2605.30621

### 11.4 Diagnose before mutating the harness (research update)
When an agent fails, “rewrite the system prompt” is a poor default. A failed trajectory may originate in a context-selection error, a malformed tool contract, a permission denial, a stale memory item, an execution-environment issue, a missing verifier, or a model limitation. Broad changes can mask the cause and create regressions.

The more mature loop is:

```text
trace → localize the responsible step and harness layer
      → state a falsifiable diagnosis
      → apply a scoped patch
      → evaluate against target failures and regression suite
      → promote, revert, or continue investigating
```

- **Paper:** *From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws (HarnessFix).* Uses trace-grounded diagnosis, a harness-aware intermediate representation, and scoped repair operators rather than broad blind mutation. https://arxiv.org/abs/2606.06324
- **Paper:** *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.* Frames each harness edit as a falsifiable contract, supported by component, experience, and decision observability. https://arxiv.org/abs/2604.25850
- **Paper:** *Retrospective Harness Optimization: Improving LLM Agents via Self-Preference over Trajectory Rollouts.* Explores selecting and improving harness updates from prior trajectories without a separately labeled validation set; study it together with SEAGym, which explains why independent frozen evaluation still matters. https://arxiv.org/abs/2606.05922

---

# PART IV — THE AUTONOMOUS FUTURE: SELF-GENERATING SKILLS, TOOLS & AGENTS

> **The thesis you've identified.** The two papers you uploaded crystallise the direction the entire field is converging on. The Skill-MAS paper shows that a frozen frontier LLM can *evolve its own orchestration strategy* (the Meta-Skill) through a closed optimization loop — without any gradient updates, without human-written rules. The Auto-SKILL.md paper shows that skill files can be *mined from interaction trajectories* rather than hand-authored. Together they point at a world where agents don't just *use* skills and tools — they *discover, design, and write* them. The human role shifts from "write the skill" to "set the goal, approve the new capability, maintain the sandbox." This is not a distant research horizon; it is already happening in production (Devin built tools it later used on itself; DGM improved from 20% to 50% on SWE-bench by rewriting its own code). This Part maps that frontier.

---

## 12. The Three-Stage Evolution: Tool User → Tool Maker → Self-Improving System

### 12.1 The paradigm shift: from static tool user to autonomous tool maker
The fundamental arc of agent capability is: (1) agents that *use* pre-built tools (fixed toolsets, human-written APIs), (2) agents that *create* tools on demand when their toolset is insufficient, (3) agents that *evolve* their own architecture, prompts, workflows, and skills over time — without weight updates. The bottleneck has always been the fixed toolset. The research frontier is removing that constraint entirely.

- **Survey (essential reading):** *A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence* (TMLR 2026) — the comprehensive map of the entire self-evolution space across model-centric, environment-centric, and co-evolution approaches. https://arxiv.org/abs/2507.21046
- **Survey repo:** Awesome-Self-Evolving-Agents — curated paper list maintained live. https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents
- **Blog (practitioner framing):** *Self-Improving AI Agents: The 2026 Guide* — o-mega.ai (Devin, DGM, Karpathy's autoresearch, the SWE-bench arc from 13% to 59%+). https://o-mega.ai/articles/self-improving-ai-agents-the-2026-guide

### 12.2 What the uploaded papers prove — and why they belong here
**Skill-MAS** (arXiv:2606.18837, HKUST/Ant Group, June 2026) solves the dilemma between inference-time MAS (high capability, no learning) and training-time MAS (learns but capped by small models). The third path: treat the Meta-agent's orchestration strategy itself as an evolvable SKILL.md. A closed loop of Multi-Trajectory Rollout + Selective Reflection refines the Meta-Skill across rounds. The evolved skills are transferable across LLMs and unseen tasks — proving that *the skill, not the weight update, is the unit of learning*.

**Auto-SKILL.md** (arXiv:2606.20363, MIT/Harvard, NeurIPS 2026) asks whether skill files can be mined from GUI interaction trajectories rather than hand-authored. The honest result: trajectory mining produces readable source-domain structure (5/8 clusters ≥0.95 purity), but the current pipeline doesn't yet beat trivial frequency baselines on cross-domain transfer. This is valuable precisely because it is a *diagnostic study* — it tells you exactly which parts of automated skill generation are solved (readable structure extraction) and which remain open (reliable cross-domain transfer).

Together, these papers define the two frontiers: *skill evolution from task experience* (Skill-MAS) and *skill discovery from interaction data* (Auto-SKILL.md). The curriculum implications: you need to understand both.

- **Paper:** *Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems* — Lin, Yang, Qin (HKUST/Ant Group), arXiv:2606.18837. https://arxiv.org/abs/2606.18837
- **Paper:** *Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining* — Hao, Li (MIT/Harvard), arXiv:2606.20363. https://arxiv.org/abs/2606.20363

### 12.3 A critical distinction: harness updating ≠ harness benefit (new research)
Self-evolving-agent papers can create an overly simple story: a strong agent reflects on trajectories, writes a better skill/prompt/tool workflow, and the next agent performs better. Recent evidence suggests there are two distinct capabilities:

1. **Harness updating:** producing an artifact that is objectively useful when evaluated as a harness change.
2. **Harness benefit:** allowing a downstream task-solving agent to retrieve, understand, activate, and follow that artifact at the right moment.

The two are not guaranteed to move together. A generated `SKILL.md` may be readable, technically sensible, and even improve a controlled evaluator, yet still fail in deployment because the receiving agent does not select it, misapplies it, or is distracted by it. The practitioner consequence is that every evolving resource requires two tests:

```text
Is this artifact good?
AND
Does the deployed agent use it correctly on new tasks?
```

- **Paper (essential calibration):** *Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents.* https://arxiv.org/abs/2605.30621
- **Cross-reference:** §11.3 for frozen validation, replay, transfer, and cost evaluation; §16.3 for the related Auto-SKILL.md lesson that readable structure does not automatically transfer.

---

## 13. Agents That Create Their Own Tools (LATM → ToolMaker → Tool-R0)

### 13.1 LATM — the foundational "tool maker" pattern
The closed-loop framework where one LLM acts as **tool maker** (crafts reusable Python tools for a class of tasks) and a cheaper LLM acts as **tool user** (applies those tools). Tools are cached via APIs and reused across requests. This decouples the cost of creation from the cost of execution — the frontier model is only called once per tool type, not per task. Every subsequent autonomous tool-creation system builds on this two-phase pattern.

- **Paper (essential):** *Large Language Models as Tool Makers (LATM)* — Cai et al., arXiv:2305.17126. The foundational paper. https://arxiv.org/abs/2305.17126
- **Blog:** *The LATM Pattern* — LinkedIn explainer with supply-chain application walkthrough. https://www.linkedin.com/pulse/latm-llm-tool-maker-pattern-christian-bauersachs-nh9lf

### 13.2 CREATOR — disentangling abstract tool design from concrete execution
CREATOR separates the *abstract reasoning* about what a tool should do from the *concrete implementation* of it. The agent first reasons about the general structure of a reusable function, then writes the specific code. This separation prevents conflating "what capability do I need?" with "how do I implement it?" — a clean architectural boundary that improves generalization and reuse.

- **Paper:** *CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation* — Qian et al., Tsinghua University, arXiv:2305.14318.

### 13.3 ToolMaker — converting code repositories into agent-usable tools autonomously
Given a GitHub URL and a short task description, ToolMaker autonomously installs dependencies, generates tool-wrapping code, and uses a closed-loop self-correction mechanism for debugging. Correctly implements 80% of tasks across 15 complex computational domains — substantially outperforming prior SWE agents. This is the step toward *fully autonomous scientific workflows*: the agent doesn't ask for a tool, it reads the paper, clones the repo, and wraps it.

- **Paper:** *LLM Agents Making Agent Tools (ToolMaker)* — Wölflein et al., arXiv:2502.11705. https://arxiv.org/abs/2502.11705

### 13.4 Tool-R0 — self-evolving tool-calling from zero data via self-play RL
Trains a general-purpose tool-calling agent from scratch with self-play RL, under a *zero-data assumption*. Two LLMs co-evolve: a Generator proposes challenging tasks at the Solver's competence frontier; the Solver attempts them. No human-labeled data, no hand-crafted task-solution pairs. This is the removal of the last human bottleneck in tool learning.

- **Paper:** *Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data* — Acikgoz et al., arXiv:2602.21320. https://arxiv.org/abs/2602.21320

---

## 14. Agents That Build Their Own Skill Libraries (Voyager → MUSE-Autoskill → Skill-MAS)

### 14.1 Voyager — the seminal ever-growing skill library (no gradient updates)
The first LLM-powered lifelong learning agent to build an ever-growing library of executable code skills, entirely without parameter fine-tuning. Three components: (1) an automatic curriculum that always proposes goals at the frontier of what the agent can do, (2) a skill library indexed by natural language and retrieved by semantic similarity, (3) iterative prompting with environment feedback + self-verification for skill refinement. Result: 3.3× more unique items, 15.3× faster tech-tree unlocks vs. prior SOTA — and the learned skills transfer to novel tasks in new worlds. This is the design template for every autonomous skill-accumulating agent that followed.

- **Paper (essential):** *Voyager: An Open-Ended Embodied Agent with Large Language Models* — Wang et al. (NVIDIA/Caltech), NeurIPS 2023, arXiv:2305.16291. https://arxiv.org/abs/2305.16291
- **Project site + code:** https://voyager.minedojo.org/

### 14.2 EvoSkill — co-evolutionary skill verification for multi-agent systems
Extends Voyager's skill evolution to multi-agent settings. Skills are discovered, refined, and verified through co-evolutionary feedback — agents critique each other's skills, creating a verification loop that weeds out brittle skills before they pollute the library.

- **Paper:** *EvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification* — Alzubi et al., arXiv:2604.01687. https://arxiv.org/pdf/2604.01687

### 14.3 MUSE-Autoskill — full skill lifecycle (create, memory, manage, evaluate)
Formalizes the four-stage skill lifecycle: (1) **Creation** — generate a skill from a successful trajectory, (2) **Memory** — store with metadata and retrieval index, (3) **Management** — score, prune, merge conflicting skills, (4) **Evaluation** — run on held-out tasks to validate generalization. The paper also shows cross-agent skill transfer: skills generated by one backbone model improve a different backbone. The management stage (preventing library rot and bloat) is the engineering insight most practitioners miss.

- **Paper:** *MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation* — arXiv:2605.27366. https://arxiv.org/html/2605.27366v1

### 14.4 AutoRefine — extracting reusable expertise from loop trajectories (cross-reference §0.5)
Already covered in §0.5 as a loop engineering paper, but belongs here too: successful trajectory patterns become specialist subagents; static knowledge becomes skill patterns; a maintenance mechanism scores, prunes, and merges. This is the bridge from "loop engineering" to "autonomous skill library building."
- **Paper:** *AutoRefine: From Trajectories to Reusable Expertise for Continual LLM Agent Refinement* — arXiv:2601.22758. https://arxiv.org/html/2601.22758v1

---

## 15. Agents That Rewrite Their Own Architecture (The Self-Modification Frontier)

### 15.1 Darwin Gödel Machine (DGM) — agents that improve by rewriting their own code
The most significant result in autonomous self-improvement to date. DGM is initialized with one coding agent and a coding benchmark. It iteratively: proposes modifications to its own Python codebase (prompts, workflows, tools), implements them, evaluates on the benchmark, adds all variants to an archive (Darwin's open-ended exploration), and selects parents for the next round. Result: SWE-bench performance improved from 20.0% to 50.0% *automatically* — the agent discovered better code-editing tools, long-context management strategies, and peer-review mechanisms *for itself*. Done with sandboxing and human oversight throughout. This is the closest thing to a self-improving system in production today.

- **Paper (essential):** *Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents* — Zhang et al. (Sakana AI + UBC), arXiv:2505.22954. https://arxiv.org/abs/2505.22954
- **Repo:** https://github.com/jennyzzt/dgm
- **Blog:** Sakana AI announcement. https://sakana.ai/dgm/

### 15.2 Self-coding information systems — agents that redeploy their own source code at runtime
Proposes systems where agentic AI dynamically generates, tests, and *redeploys* their own source code at runtime to reduce feature delivery time. The implication: the agent doesn't just modify its prompts or skills — it modifies the running system and hot-swaps it. This is the software engineering version of self-modification.

- **Reference:** *Toward self-coding information systems* — listed in the awesome-ai-agent-papers repo. https://github.com/VoltAgent/awesome-ai-agent-papers

### 15.3 CORAL — multi-agent systems that self-evolve via shared persistent memory
Long-running multi-agent systems where agents self-evolve through shared persistent memory and open-ended discovery. Multiple agents contribute to a growing shared knowledge base; the topology of the agent network itself evolves.

- **Reference:** *CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery* — in awesome-ai-agent-papers. https://github.com/VoltAgent/awesome-ai-agent-papers

---

## 16. What This Means for the Practitioner: The New Engineering Contract

### 16.1 The three-level autonomy ladder (and where humans still belong)
The practitioner framing from Vellum's 2026 agent guide makes the stakes concrete:

Level 1 — **AI Workflow**: makes output decisions based on natural language. Fully human-directed.
Level 2 — **Router Workflow**: chooses which tasks and tools to execute. Human defines the tool inventory.
Level 3 — **Autonomous Agent**: creates new tasks *and new tools* to execute. Human sets goals, approves new capabilities, maintains the sandbox.

Most teams in 2025 were at Level 2. The research in §13–15 is what Level 3 looks like in practice. The engineering discipline shifts: you stop writing tools and start designing the *evaluation criteria* and *safety boundaries* under which the agent is allowed to create them.

- **Blog:** *The 2026 Guide to AI Agent Workflows* — Vellum (three-level taxonomy, LATM, tool creation). https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns
- **Blog:** *Taming AI Agents: The Autonomous Workforce of 2026* — CIO (by early 2026, agents write their own tools; containment principles). https://www.cio.com/article/4064998/taming-ai-agents-the-autonomous-workforce-of-2026.html

### 16.2 The new human role: capability approver + sandbox maintainer
When agents generate their own skills and tools, the human role is no longer "write the SKILL.md" — it is:
1. **Set the goal and success criteria** (the verifier, as §0 established, remains human-defined).
2. **Approve new capability requests** — when an agent determines it needs a tool that doesn't exist, it surfaces a request (the equivalent of a PR). The human reviews, approves, or rejects.
3. **Maintain the sandbox and blast radius** — as generated tools compound, the attack surface grows. §5 (sandboxing) and §10 (guardrails) become more critical, not less, as autonomy increases.
4. **Manage library rot** — the skill library needs the same maintenance discipline as a codebase: scoring, pruning, merging, deprecating (§14.3 MUSE-Autoskill lifecycle).

- **Survey (§3.3 — tools):** *A Survey of Self-Evolving Agents* (autonomous discovery, mastery, management — the three fronts). https://arxiv.org/abs/2507.21046
- **Cross-reference:** §5 (Environment, Sandboxing & Safety), §10 (Guardrail Middleware), §0.6 (Loop failure modes).

### 16.3 The honest negative result: readability ≠ transferability (the Auto-SKILL.md lesson)
The Auto-SKILL.md paper's diagnostic finding is the most important calibration for practitioners excited about automated skill generation: *trajectory mining can produce readable skill structure, but the current pipeline doesn't beat trivial frequency baselines on cross-domain transfer.* The mined skills are inspectable; they are not yet reliable. This means: automated skill generation is currently most valuable as a *first draft* that humans review and refine, not as a fully autonomous pipeline. The bottleneck is the reward model and the ordering-unaware segment representation — both are active research problems.

Treat automated skill generation like AI-generated code: useful, often correct in structure, requires human review before production deployment.

- **Paper:** Auto-SKILL.md (§12.2). The negative result section is the most important read. https://arxiv.org/abs/2606.20363
- **Cross-reference:** §0.6 (failure modes of loops), §11.2 (eval-driven improvement).

### 16.4 The missing operational layer: versioned resources, promotion, and rollback
As prompts, skills, tools, memory schemas, environments, and subagents become editable resources, they need lifecycle management just as code does. A generated capability should have an identifier, owner, version, provenance, evaluation record, scope of permissions, promotion status, and rollback path.

Treat a new generated tool or skill as a pull request:

```text
proposal
→ static review and permission review
→ sandbox evaluation
→ frozen regression suite
→ staged promotion
→ trace-backed monitoring
→ rollback or deprecation if it harms reliability
```

This prevents an autonomous agent from silently accumulating an opaque pile of prompts, tool wrappers, and memory artifacts whose effects nobody can reconstruct.

- **Paper:** *Autogenesis: A Self-Evolving Agent Protocol.* Models prompts, agents, tools, environments, memory, and outputs as versioned resources with explicit lifecycle state, plus propose/assess/commit/rollback interfaces. https://arxiv.org/abs/2604.15034
- **Survey:** *Externalization in LLM Agents.* Useful for seeing memory, skills, protocols, and harnesses as coupled forms of externalized capability rather than unrelated features. https://arxiv.org/abs/2604.08224
- **Cross-reference:** §5 for the sandbox boundary, §10 for approval interrupts, and §11.3 for release-grade evaluation.

---

# PART III — THE LEARNING PLAN

## How to actually learn this (suggested 8-week path)

**Week 0 — Loop Engineering methodology.** Read §0.1–0.4. Read Tosea.ai and AI Builder Club. Deliverable: for one repetitive task, write down the trigger, state check, exit condition, and open/closed choice — before writing any code.

**Week 1 — Mental model.** Read the expanded §1.1–1.3. Read Anthropic *Building Effective Agents* and OpenAI *Unrolling the Codex agent loop* cover to cover. Deliverable: write the `while(tool_call)` loop from memory in ~30 lines of pseudocode, draw the three-plane harness architecture, and name one failure mode plus one observable signal for each plane.

**Week 2 — Read the source.** Clone `openai/codex`, skim `codex-rs/`, read claude-code-ultimate-guide architecture.md. Deliverable: one-page diff of how Codex and Claude Code handle tools, memory, and sandboxing.

**Week 3 — Repo scaffold.** Read §2. Write a real ~100-line AGENTS.md backed by a `docs/` system-of-record. Read *Harness engineering: leveraging Codex* twice.

**Week 4 — Long-running + safety.** Read §4 and §5. Build the initializer/coder pattern. Add one real hook and one sandbox boundary. Read *Managed Agents* for the assumptions-decay lesson.

**Week 5 — Middleware.** Read §7–8, including §8.4. Build in LangChain/LangGraph: a summarization middleware + a token-monitor trigger + one interrupt/checkpoint before a destructive tool. Add a trace field explaining why each context item was included, compressed, or dropped.

**Week 6 — Guardrails, routing, observability.** Read §9–11, including the new §11.3–11.4. Add a model-routing rule, an input/output guardrail, structured logging, a small frozen regression set, and one failure-diagnosis template that distinguishes context, tool, policy, environment, verifier, and model failures.

**Week 7 — Capstone: build a full loop.** Design and ship a closed loop end to end: pick a real repetitive task, define a measurable verifier, wire a trigger, build orchestrator + two specialist sub-agents, add a human-escalation checkpoint, assemble all five harness layers with a CI gate. Write up where the verifier was the bottleneck.

**Week 8 — The autonomous future.** Read §12–16, especially §12.3 and §16.4. Read the Skill-MAS paper (§12.2), *Harness Updating Is Not Harness Benefit* (§12.3), and the DGM paper (§15.1). Read the Self-Evolving Agents survey (§12.1) §3.3 on tools. **Reflection deliverable:** for your Week 7 capstone loop, answer: (a) which skills could be *discovered* from trajectories rather than hand-written, (b) which tools could be *generated* rather than pre-built, (c) what evaluation criteria would you need in place to safely allow the agent to create a new tool, (d) what sandbox boundary would prevent a runaway self-modification loop, and (e) how would you prove the deployed agent actually benefits from a new skill rather than merely storing it?

## The shortlist (if you only read 15 things)
1. Tosea.ai — *What Is Loop Engineering?* (the four-era lineage)
2. AI Builder Club — *Loop Engineering Guide 2026* (verifier-as-bottleneck)
3. arXiv 2210.03629 — *ReAct* (the atomic loop)
4. arXiv 2303.17651 — *Self-Refine* (the self-improvement loop)
5. arXiv 2303.11366 — *Reflexion* (the memory loop)
6. Anthropic — *Building Effective Agents* (foundations + patterns)
7. OpenAI — *Unrolling the Codex agent loop* (the loop, precisely)
8. OpenAI — *Harness engineering: leveraging Codex* (the repo-scaffold masterclass)
9. Anthropic — *Effective context engineering for AI agents*
10. Anthropic — *Effective harnesses for long-running agents*
11. Anthropic — *Scaling Managed Agents*
12. LangChain — *Context engineering in agents* (the middleware model)
13. arXiv 2305.17126 — *LATM* (agents that make tools)
14. arXiv 2305.16291 — *Voyager* (the ever-growing skill library — the design template)
15. arXiv 2505.22954 — *Darwin Gödel Machine* (agents that rewrite their own code — the frontier)

## 2026 research addendum (new references added in this update)
Read these after the core 15-item shortlist. They sharpen the curriculum’s weakest point: how to evaluate, diagnose, and safely deploy an evolving harness.

1. *Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems* — unified survey of harness interfaces, mechanisms, scaling, and open challenges. https://arxiv.org/abs/2605.18747
2. *Externalization in LLM Agents* — unified view of memory, skills, protocols, and harness engineering. https://arxiv.org/abs/2604.08224
3. *Harnesses for Inference-Time Alignment over Execution Trajectories* — why more scaffolding can hurt; study over-decomposition and over-pruning. https://arxiv.org/abs/2605.21516
4. *A Benchmark for Context Retrieval in Coding Agents (ContextBench)* — process-level retrieval precision, recall, and efficiency. https://arxiv.org/abs/2602.05892
5. *Meta Context Engineering via Agentic Skill Evolution* — co-evolves context-management skills and context artifacts. https://arxiv.org/abs/2601.21557
6. *InfiAgent* — externalized file-centric state for bounded long-horizon reasoning. https://arxiv.org/abs/2601.03204
7. *LongSeeker* — context operations as explicit agent actions: Skip, Compress, Rollback, Snippet, Delete. https://arxiv.org/abs/2605.05191
8. *SEAGym* — a release-like evaluation environment for self-evolving harnesses, including frozen validation, transfer, replay, cost, and snapshots. https://arxiv.org/abs/2606.17546
9. *HarnessFix* — trace-guided diagnosis and scoped repair of harness flaws. https://arxiv.org/abs/2606.06324
10. *Agentic Harness Engineering* — observability-driven automatic harness evolution using falsifiable edit contracts. https://arxiv.org/abs/2604.25850
11. *Harness Updating Is Not Harness Benefit* — distinguish creating useful updates from a deployed agent actually benefiting from them. https://arxiv.org/abs/2605.30621
12. *Autogenesis: A Self-Evolving Agent Protocol* — lifecycle, versioning, provenance, promotion, and rollback for evolving agent resources. https://arxiv.org/abs/2604.15034

## A note on currency
Codex models move fast (GPT-5.2-Codex → GPT-5.3-Codex shipped within this window) and harness internals change weekly. Always check the live **Codex Changelog** (https://developers.openai.com/codex/changelog), the **Anthropic Engineering** hub (https://www.anthropic.com/engineering), and **docs.claude.com** for the current state before relying on any specific version detail above.

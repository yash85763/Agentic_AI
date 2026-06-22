# Loop Engineering, Harness Engineering & Agentic Middleware Engineering
### A Complete Learning Curriculum (Table of Contents + Resources per Concept)

> **The one-line thesis:** `Agent = Model + Harness`. The model is roughly fixed; almost everything you can actually engineer lives in the layer *around* it. LangChain's DeepAgent jumped **13.7 points on Terminal-Bench 2.0 (52.8% → 66.5%) by changing only the harness — same model**. That gap is the entire discipline.
>
> **How to read this document:** Concepts are ordered as a learning path. Each concept has (a) a one-paragraph "what it is," and (b) the canonical blog/paper/repo to learn it from. **Part 0** is **Loop Engineering** (the practitioner methodology — designing the *system* that prompts agents, not the prompts themselves). **Part I** is **Harness Engineering** (the agent runtime + the repo/context scaffold around a coding agent like Claude Code or Codex). **Part II** is **Agentic Middleware Engineering** (the programmable interception layer *inside* the loop — context, routing, guardrails, observability). **Part III** is the study plan. Loop Engineering is *why* you build the harness; Harness Engineering is *what* you build; Middleware Engineering is *how* the cross-cutting concerns get implemented.

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

### 1.1 Why the harness is the job
The counterintuitive core finding: model swaps matter less than harness design. 84% of developers use AI tools but only ~29% trust the output; closing that trust gap is harness work. Start here to internalize *why* this discipline exists.
- **Blog:** *Agentic Harness Engineering: LLMs as the New OS* — Decoding AI. https://www.decodingai.com/p/agentic-harness-engineering
- **Blog (reading-path overview):** *The Complete Claude Code Harness Engineering Guide (5 Layers, 8 Deep-Dives)* — DEV. https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j
- **Definitional companion:** *Claude Code Harness and Environment Engineering* — hidekazu-konishi.com (splits the discipline into *harness engineering* = the runtime, and *environment engineering* = bounding the world). https://hidekazu-konishi.com/entry/claude_code_harness_and_environment_engineering_guide.html

### 1.2 The agentic loop (the irreducible core)
Every coding agent is a surprisingly simple `while (tool_calls) { call model → run tools → append results }` loop. "Agentic" behavior *emerges* from repetition, not from a planning engine. Understand this before anything else.
- **Primary (OpenAI):** *Unrolling the Codex agent loop* — exact request construction, message roles, turn semantics. https://openai.com/index/unrolling-the-codex-agent-loop/
- **Primary (Anthropic foundations):** *Building Effective Agents* — workflows vs. agents, when to use each, orchestrator-workers, evaluator-optimizer. https://www.anthropic.com/research/building-effective-agents
- **Deep dive (cross-tool):** *Inside the Agent Harness: How Codex and Claude Code Actually Work* — turn.rs pseudocode, compaction, token heuristics. https://medium.com/jonathans-musings/inside-the-agent-harness-how-codex-and-claude-code-actually-work-63593e26c176
- **Repo (read the source):** `openai/codex` — the open-sourced Codex CLI (Rust); read `codex-rs/`. https://github.com/openai/codex

### 1.3 The architecture of a real harness (Claude Code, dissected)
Claude Code runs a `while(tool_call)` loop with ~8 core tools (Bash, Read, Edit, Write, Grep, Glob, Task/sub-agents, TodoWrite) — no RAG, no DAG. Anthropic dropped semantic embedding search in favor of grep-based agentic search. Study this as the reference implementation.
- **Repo/guide:** *claude-code-ultimate-guide* — architecture.md is a verified technical dissection. https://github.com/FlorianBruniaux/claude-code-ultimate-guide/blob/main/guide/core/architecture.md
- **Comparative source analysis:** *OpenAI Codex CLI Architecture and Multi-Runtime Agent Patterns* — Zylos (bubblewrap sandbox, execpolicy DSL, two-phase memory, app-server JSON-RPC; side-by-side with Claude Code). https://zylos.ai/research/2026-03-26-openai-codex-cli-architecture-multi-runtime-patterns/

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

---

# PART III — THE LEARNING PLAN

## How to actually learn this (suggested 7-week path)

**Week 0 — Loop Engineering methodology.** Read §0.1–0.4 fully. Then read the two primary blog-level resources: Tosea.ai (the lineage) and AI Builder Club (verifier-as-bottleneck checklist). Watch the two videos referenced in §0. Deliverable: for one repetitive task you do manually, write down (a) the trigger, (b) the state check, (c) the exit condition, and (d) whether it should be open or closed — *before* writing a single line of code.

**Week 1 — Mental model.** Read §1.1–1.3 end to end. Then read the two foundational primaries cover to cover: Anthropic *Building Effective Agents* and OpenAI *Unrolling the Codex agent loop*. Deliverable: write the `while(tool_call)` loop from memory in ~30 lines of pseudocode.

**Week 2 — Read the source.** Clone `openai/codex` and skim `codex-rs/`; read the claude-code-ultimate-guide architecture.md alongside it. Deliverable: a one-page diff comparing how Codex and Claude Code handle tools, memory, and sandboxing (use the Zylos analysis as a key).

**Week 3 — Repo scaffold.** Read §2 fully. Write a real ~100-line AGENTS.md for one of your own repos using the "table-of-contents not encyclopedia" principle, backed by a `docs/` system-of-record. Read the OpenAI *Harness engineering* case study twice.

**Week 4 — Long-running + safety.** Read §4 and §5. Build the initializer/coder pattern from Anthropic's long-running-agents quickstart. Add one real hook and one sandbox boundary. Read *Managed Agents* for the "assumptions decay" lesson.

**Week 5 — Middleware.** Read §7–8 and build in LangChain/LangGraph: a summarization middleware + a token-monitor trigger + one interrupt/checkpoint before a destructive tool. Read the LangChain context-engineering docs as you go.

**Week 6 — Guardrails, routing, observability.** Read §9–11. Add a model-routing rule, an input/output guardrail, and structured logging.

**Week 7 — Capstone: build a full loop.** Design and ship a closed loop end to end: pick a real repetitive task, define a measurable verifier, choose open or closed, wire a trigger (cron or webhook), build the orchestrator + at least two specialist sub-agents, add a human-escalation checkpoint, and assemble all five harness layers (Memory → Tools → Permissions → Hooks → Observability) with a CI gate. Write up where the verifier was the bottleneck and where harness assumptions had to be revised.

## The shortlist (if you only read 12 things)
1. Tosea.ai — *What Is Loop Engineering?* (the four-era lineage + definition)
2. AI Builder Club — *Loop Engineering Guide 2026* (verifier-as-bottleneck, open/closed decision rule)
3. arXiv 2210.03629 — *ReAct* (the atomic loop: thought-action-observation)
4. arXiv 2303.17651 — *Self-Refine* (the self-improvement loop)
5. arXiv 2303.11366 — *Reflexion* (the memory loop: verbal RL without weight updates)
6. Anthropic — *Building Effective Agents* (foundations + patterns)
7. OpenAI — *Unrolling the Codex agent loop* (the loop, precisely)
8. OpenAI — *Harness engineering: leveraging Codex in an agent-first world* (the repo-scaffold masterclass)
9. Anthropic — *Effective context engineering for AI agents*
10. Anthropic — *Effective harnesses for long-running agents*
11. Anthropic — *Scaling Managed Agents: Decoupling the brain from the hands*
12. LangChain — *Context engineering in agents* (the middleware model)

## A note on currency
Codex models move fast (GPT-5.2-Codex → GPT-5.3-Codex shipped within this window) and harness internals change weekly. Always check the live **Codex Changelog** (https://developers.openai.com/codex/changelog), the **Anthropic Engineering** hub (https://www.anthropic.com/engineering), and **docs.claude.com** for the current state before relying on any specific version detail above.

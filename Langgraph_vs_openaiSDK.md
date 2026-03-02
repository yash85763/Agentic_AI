Here’s a fresh, research-backed “reading + implementation” pack (late-2025 → early-2026) that maps directly onto the core pillars of Jarvis. I’m prioritizing papers that (a) are very recent, (b) are immediately actionable, and (c) cover the failure modes that kill agent products in the wild (latency, cost, security, long-horizon reliability).

⸻

A) Long-horizon “Software Factory” that finishes (and doesn’t burn money)

1) BOAD (Dec 2025) — learn the best org chart, don’t guess it

Why it matters: it formalizes discovering a good hierarchy (orchestrator + subagents) using multi-armed bandits, and reports strong results on SWE-bench variants (incl. OOD SWE-bench-Live).
How you use it in Jarvis: start with your PM→TL→Dev→QA→Sec flow, then learn routing: which sub-agent sequences work best for bugfix vs feature vs refactor, and when to invoke expensive steps.  ￼

2) SWE-Replay (Jan 2026) — trajectory replay/branching to cut cost without losing quality

Why it matters: test-time scaling is expensive if you “resample from scratch.” SWE-Replay reuses prior trajectories and branches at meaningful intermediate states.
How you use it: build a debug-trajectory cache keyed by repo state + failure signature + test output; when similar failures happen, Jarvis “replays” the successful exploration path instead of re-exploring.  ￼

3) SWE-Universe (Feb 2026) — millions of verifiable SWE environments + “in-loop hacking detection”

Why it matters: it’s about scaling verifiable environments from GitHub PRs, dealing with build yield, verifiers, cost, and explicitly mentions self-verification + hacking detection during environment construction.
How you use it: (later) for Jarvis “learning”: generate a large pool of realistic, verifiable tasks and train internal policies (routing, patching, tool-usage) safely in sandboxes.  ￼

⸻

B) “Body” (computer-use / tool-use) that is reliable and fast

4) OSWorld-Human (Jun 2025) — latency is the real bottleneck, not accuracy

Why it matters: shows planning/reflection dominates latency, and steps get slower over time as history grows; provides human trajectories to measure “wasted steps.”
How you use it: make efficiency a product KPI (steps-over-human, time-per-step, tool-call rate), enforce state compression, and cap reflection tokens.  ￼

5) OSWorld-MCP (Oct 2025) — tool invocation is now benchmarked

Why it matters: evaluates tool invocation + GUI together; finds tools help but models often under-invoke them.
How you use it: treat “when to call tools” as a first-class policy with budgets + forced tool checks at gates (e.g., always run tests before “done”).  ￼

6) Agent S2 (Apr 2025) — split generalist vs specialists for grounding + planning

Why it matters: proposes compositional generalist/specialist separation (better grounding + hierarchical planning).
How you use it: implement Jarvis “Body” as three roles: (1) perception/grounding (VLM), (2) planner, (3) executor, with cached perception and minimal re-reads.  ￼

7) OpenAI CUA (Operator baseline) — what “operator-class” looks like

Why it matters: gives a concrete reference for OSWorld/WebArena/WebVoyager performance claims and constraints.
How you use it: use it as your north-star baseline for computer-use flows, but keep Jarvis local-first + permissioned.  ￼

⸻

C) Graph Memory & multimodal “mind map” that stays true (with provenance + time)

8) Query-Driven Multimodal GraphRAG (ACL Findings 2025) — dynamic query-local graphs from multimodal evidence

Why it matters: directly supports your “global graph + query-local subgraph” approach, and treats multimodal inputs (screenshots/diagrams) as first-class.
How you use it: don’t precompute everything forever; build/refresh query-local subgraphs on demand, and attach every node/edge to evidence (span/page/bbox).  ￼

9) Zep / Graphiti (Jan 2025) — temporal knowledge graph for agent memory

Why it matters: explicitly focuses on temporally-aware KG memory (historical relationships), evaluated on long-memory tasks.
How you use it: your memory schema should include valid_from/valid_to, “decision versions,” and “claim conflict sets,” so Jarvis can answer “what changed since last week?” not just “what is true.”  ￼

⸻

D) Security (this is existential for Jarvis)

10) Securing MCP (Dec 2025) — tool descriptor attacks: poisoning, shadowing, rug pulls

Why it matters: shows the attack surface isn’t only prompts—tool metadata can be weaponized. Proposes signing + semantic vetting + runtime guardrails.
How you use it: treat MCP servers as untrusted: signed manifests, descriptor integrity, allow-lists, and anomaly blocking at runtime.  ￼

11) AgentSentry (Feb 26, 2026) — inference-time defense against indirect prompt injection (multi-turn)

Why it matters: tackles multi-turn “silent takeover” via tool outputs / retrieved context, using temporal causal diagnostics + context purification.
How you use it: add a “security gate” between tool-return → planner: detect takeover signals and purify context while preserving legitimate evidence.  ￼

12) MemoryGraft (Dec 18, 2025) — persistent memory poisoning via “poisoned experience retrieval”

Why it matters: your long-term memory is a new trust boundary; attackers can implant “successful-looking” procedures that get reused later.
How you use it: separate fact memory vs procedure memory, sign/attest trusted procedures, and run “memory quarantine” for new experiences until validated by tests + policy.  ￼

⸻

E) Practical ecosystem leverage: MCP packaging & tool onboarding

13) Claude Desktop Extensions (.mcpb) (Jun 26, 2025) — one-click tool server installation pattern

Why it matters: shows how fast tool ecosystems grow when integration is standardized and install is trivial.
How you use it: adopt MCP compatibility early, but keep Jarvis’s runtime least-privilege + audit-first.  ￼

⸻

What to implement first (based on the above evidence)
	1.	Verification-gated “Software Factory” loop + trajectory replay

	•	BOAD-inspired modular hierarchy + logging now; replay cache next.  ￼

	2.	Time-aware graph memory

	•	Start with temporal KG primitives (validity windows + provenance) + query-local subgraphs for speed.  ￼

	3.	Security hardening before “tool ecosystem scale”

	•	MCP descriptor signing/vetting + IPI defense + memory poisoning defenses.  ￼

	4.	Latency as a KPI

	•	OSWorld-Human-style metrics from day 1 (steps, time, tool calls, reflection budget).  ￼

⸻

If you want, I can turn this into a concrete “Jarvis research-to-roadmap spec”: a table of paper → feature → architecture decision → measurable KPI → MVP cut (so it’s investor-ready and engineering-ready).
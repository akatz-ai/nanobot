---
schema_version: 1
id: mem-phase3-retrieval
title: "Memory Phase 3 \u2014 Retrieval overhaul + MEMORY.md transition"
status: active
priority: 2
tags:
- memory,retrieval
created_at: '2026-03-04T08:16:41Z'
updated_at: '2026-03-04T08:41:16Z'
---

## Tasks
- [ ] mem-phase3-retrieval.0 Replace scope model — replace agent_id + peer_key + visibility with owner_agent_id + scope (session|agent|shared|global) + scope_id throughout store and retriever {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.1 Implement retrieval pools — pinned seed pool (identity, core preferences) + dynamic pool (session episodic + agent semantic memories) with scope-aware reranking {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.2 Implement decay-aware scoring — combine semantic relevance, entity overlap, scope bonus, importance, recency, reinforcement count, decay class into retrieval ranking {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.3 Compute retrieval headroom from assembled base prompt — budget retrieval tokens against actual remaining context, not hardcoded limits {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.4 Inject retrieved memory as separate late system message — not mixed into static system prompt, cache-friendly placement {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.5 Parse existing MEMORY.md into pinned v2 seed records — map sections to record_types (Identity→identity, Preferences→preference, Decisions→decision, Facts→fact, Projects→goal) {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.7 Stop auto-rewriting MEMORY.md from extraction — remove HybridMemoryManager file write-through in graph_first mode {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.8 Update prompt contract — remove instructions telling agent to write to MEMORY.md; update memory_save tool to write to graph not file {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.9 Remove daily-history injection in graph_first mode — retrieval replaces it {deps=[mem-phase2-extraction.9]}
- [ ] mem-phase3-retrieval.10 Add tests — scope filtering correctness, pool reranking, MEMORY.md import idempotency, no cross-session leakage in retrieval, prompt budget compliance {deps=[mem-phase2-extraction.9]}

## Notes
Retrieval overhaul and MEMORY.md transition. Start with 2 pools (pinned + dynamic) not 4 — add more as we learn. Clean scope model replaces broken agent_id/peer_key/visibility. MEMORY.md becomes a one-time import into pinned seed records then stops being auto-updated. Prompt contract updated so agents dont know about storage backends. Reference: agents/shared/oracle/nanobot-memory-redesign-audit-v6.md

Simplified: no separate archive step for MEMORY.md — just back it up as part of the seed import task. Cut over retrieval directly, no gradual rollout needed.

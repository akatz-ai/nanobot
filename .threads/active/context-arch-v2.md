---
schema_version: 1
id: context-arch-v2
title: "Context Architecture V2 \u2014 Cache-optimized layout with inline per-turn\
  \ context"
status: active
priority: 1
created_at: '2026-03-07T05:20:20Z'
updated_at: '2026-03-07T05:20:39Z'
---

## Tasks
- [ ] context-arch-v2.0 Remove daily history injection from ContextBuilder
- [ ] context-arch-v2.1 Move session info from system message to user message prepend {deps=[context-arch-v2.0]}
- [ ] context-arch-v2.2 Move retrieved memory from system message to user message prepend {deps=[context-arch-v2.1]}
- [ ] context-arch-v2.3 Create _build_turn_context() method that formats [Current Session] + [Retrieved Memory] as prefix {deps=[context-arch-v2.0]}
- [ ] context-arch-v2.4 Update build_messages() to use new V2 layout: system prompt → MEMORY.md → compaction summary → history → prefixed user message {deps=[context-arch-v2.3]}
- [ ] context-arch-v2.5 Ensure compaction summary is classified as static prefix (not per-turn) {deps=[context-arch-v2.4]}
- [ ] context-arch-v2.6 Update context_log.py to capture per-user-message prepended context instead of system-level injection {deps=[context-arch-v2.4]}
- [ ] context-arch-v2.7 Update all loop.py call sites that pass memory_context to build_messages() {deps=[context-arch-v2.4]}
- [ ] context-arch-v2.8 Cache stability tests: verify consecutive build_messages() calls with different user messages produce identical system prefix {deps=[context-arch-v2.7]}
- [ ] context-arch-v2.9 Backward compat tests: old sessions with system messages for session info/daily history still load correctly {deps=[context-arch-v2.7]}
- [ ] context-arch-v2.10 Update existing tests that assert on message structure (system message count, order, content) {deps=[context-arch-v2.7]}

## Notes
Full spec: /data/projects/nanobot/docs/context-architecture-v2-spec.md

Key points:
- System messages (prompt, MEMORY.md, compaction summary) form a STABLE cached prefix
- Session info + retrieved memory are prepended INLINE to each user message
- Daily history is REMOVED (replaced by MEMORY.md + graph recall + compaction summary)
- Only tool pruning and full compaction mutate existing messages
- Retrieved memory is NOT repeated — each user msg gets its own retrieval, stored once
- User messages in session JSONL include the prepended context (no re-retrieval needed on replay)

---
schema_version: 1
id: compaction-integration
title: "Compaction integration \u2014 wire new compaction into agent loop, replace\
  \ broken consolidation"
status: active
priority: 1
created_at: '2026-02-28T08:59:54Z'
updated_at: '2026-02-28T21:40:20Z'
---

## Tasks
- [x] compaction-integration.0 Implement compact_session() orchestrator function — ties together: should_compact() check → find_cut_point() → generate_compaction_summary() (with iterative update if previous exists) → extract_file_ops() → session.append_compaction(). Returns CompactionResult or None.
- [x] compaction-integration.1 Replace _process_message compaction block in loop.py — remove the 80-line inline compaction block. Replace with: if should_compact() → compact_session() → log event. Remove context_anchor manipulation, remove continuity snapshot. {deps=[compaction-integration.0]}
- [x] compaction-integration.2 Update context.build_messages() — remove continuity_context parameter. Instead, check session for CompactionEntry and inject summary as system message labeled 'Session Context Summary' before history messages. {deps=[compaction-integration.1]}
- [x] compaction-integration.3 Wire memory extraction into compaction flow — after generating the structured summary, also run the existing hybrid consolidator on the compacted messages to extract facts to MEMORY.md/graph. The summary handles continuity, extraction handles long-term memory. Both run. {deps=[compaction-integration.2]}
- [x] compaction-integration.4 Update /new command handler — when archiving all messages, generate a final compaction summary before clearing. This ensures facts are extracted even on manual session reset. {deps=[compaction-integration.3]}
- [x] compaction-integration.5 Remove continuity_context hack — delete _snapshot_continuity_context(), remove continuity_context from session.metadata, remove continuity_expires_at_message_count TTL logic, remove _clear_expired_continuity_context(). {deps=[compaction-integration.4]}
- [x] compaction-integration.6 Simplify session cursors — context_anchor metadata is replaced by CompactionEntry.first_kept_index. last_consolidated remains for tracking which messages have had facts extracted (separate concern from context windowing). {deps=[compaction-integration.5]}
- [x] compaction-integration.7 Update compaction logging — CompactionEvent should record the new structured data: summary length, file_ops, cut point details, whether it was an iterative update. {deps=[compaction-integration.6]}
- [x] compaction-integration.8 Add test: full compaction flow — session with 50+ messages triggers compaction, produces CompactionEntry, subsequent get_history() returns summary + recent messages {deps=[compaction-integration.7]}
- [x] compaction-integration.9 Add test: iterative compaction — two compactions in sequence, second uses update prompt and preserves first summary's info {deps=[compaction-integration.8]}
- [x] compaction-integration.10 Add test: /new command archives and extracts before clearing {deps=[compaction-integration.9]}
- [x] compaction-integration.11 Add test: compaction + memory extraction both run (summary for continuity, facts for MEMORY.md) {deps=[compaction-integration.10]}
- [x] compaction-integration.12 DEPENDENCY: compaction-entry, compaction-summary, and compaction-cut-points must all be complete {deps=[compaction-cut-points.7]}
- [x] compaction-integration.13 Resolve summary injection conflict — remove task to inject in build_messages(). Summary is injected by get_history() only (decided in Phase 1). Update build_messages() to remove continuity_context parameter but do NOT add compaction summary injection there.
- [x] compaction-integration.14 Add CompactionPlan dataclass — fields: summary_start, summary_end (range to summarize), first_kept_index (where to cut), extract_start, extract_end (range for memory extraction). Separates summary window from extraction window explicitly.
- [x] compaction-integration.15 Update _consolidate_memory() signature — accept optional explicit extraction range (start_index, end_index) instead of deriving from context_anchor/keep_count. When range is provided, use it. When not provided (legacy path), fall back to existing behavior.
- [x] compaction-integration.16 Keep _plan_hybrid_batches() — Codex found this solves extraction batching (token-aware batch splitting for memory extraction), NOT context windowing. Cut point logic replaces context_anchor for windowing; batch planning remains for extraction.
- [x] compaction-integration.17 Add failure policy — if summary generation succeeds but memory extraction fails: commit CompactionEntry (continuity is more important), log extraction failure, retry extraction on next compaction cycle. If summary fails: do NOT commit CompactionEntry, clear stale token state, retry on next message.
- [x] compaction-integration.18 Keep preflight/no-op logic — port _compaction_would_execute() equivalent into new orchestrator. Clear stale token state on no-op to prevent re-trigger loops. This was a hard-won fix.
- [x] compaction-integration.19 Add feature flag — config option compaction.engine: 'legacy' | 'structured' (default 'legacy' during rollout). When 'structured', use new CompactionEntry flow. When 'legacy', use existing consolidation. Allows safe rollout.
- [x] compaction-integration.20 Update dashboard schema — CompactionEvent fields change: remove context_anchor, new_context_anchor, continuity_snapshot_chars. Add summary_length, file_ops_read_count, file_ops_modified_count, is_iterative_update, cut_point_type (clean|split_turn).
- [x] compaction-integration.21 Update context_log — remove continuity_context labels/UI paths. Add compaction_summary_injected flag to turn context log.
- [x] compaction-integration.22 Migrate existing tests — update tests that assert context_anchor behavior (test_commands.py:600-644). Replace with assertions on CompactionEntry.first_kept_index.
- [x] compaction-integration.23 Add concurrency note — preserve per-session lock + global memory file lock semantics from current compaction. New orchestrator must acquire same locks.

## Notes
## Codex Audit Fixes Applied

### Fix: Summary injection conflict resolved
Phase 1 task .3 and Phase 4 task .2 both specified summary injection in different places. RESOLVED: injection happens in Session.get_history() only. build_messages() just removes continuity_context parameter, does NOT add compaction injection.

### Fix: Explicit extraction boundaries
_consolidate_memory() currently derives extraction range from context_anchor/keep_count. New CompactionPlan dataclass provides explicit summary_start/end and extract_start/end ranges. _consolidate_memory() updated to accept optional explicit range.

### Fix: Keep _plan_hybrid_batches()
Codex correctly identified that _plan_hybrid_batches() solves token-aware batch splitting for MEMORY EXTRACTION, not context windowing. Cut point logic replaces context_anchor for windowing; batch planning remains for extraction. These are separate concerns.

### Fix: Failure policy
Need explicit policy for partial failures:
- Summary OK + extraction fails → commit CompactionEntry (continuity > extraction), retry extraction later
- Summary fails → do NOT commit, clear stale tokens, retry next message
- Both fail → same as summary fails

### Fix: Feature flag for safe rollout
compaction.engine config: 'legacy' (default) | 'structured'. Allows running old and new side-by-side during testing.

### Fix: Dashboard/context-log migration
Codex found that dashboard and context_log are tightly coupled to current compaction fields. Added explicit migration tasks.

### Fix: Test migration
Existing tests assert context_anchor behavior. These must be updated, not just deleted.

### Fix: Concurrency
Current compaction uses per-session asyncio.Lock + global _memory_file_lock. New orchestrator must preserve these semantics.
## Design Context

Phase 4: Wire everything together. This is where the rubber meets the road — replacing the broken compaction in loop.py with the new system.

### Current Compaction Flow (broken)
1. Token threshold check → _should_compact_by_tokens()
2. Preflight check → _compaction_would_execute()
3. Send 'compacting...' notice to user
4. Snapshot continuity context (last 3-4 exchanges)
5. Advance context_anchor
6. Run _consolidate_memory() which either:
   a. hybrid engine: calls memory_module.hybrid.compact() — extracts facts, rewrites MEMORY.md
   b. legacy engine: calls MemoryStore.consolidate() — old flat-file extraction
7. Clear stale token readings
8. Send 'compacted' notice

### New Compaction Flow
1. should_compact() — uses actual token usage data from last LLM response
2. find_cut_point() — smart turn-aware boundary selection
3. generate_compaction_summary() — structured LLM summary (iterative if previous exists)
4. extract_file_ops() — track read/modified files
5. session.append_compaction() — store CompactionEntry in JSONL
6. Run memory extraction on compacted messages (extract facts to MEMORY.md/graph)
7. Send notice to user

### Key Architectural Change
**Separation of concerns:**
- CompactionEntry handles CONTEXT CONTINUITY (what was I doing?)
- Memory extraction handles LONG-TERM FACTS (what should I remember forever?)
- These are independent — the summary is ephemeral (replaced on next compaction), facts are permanent

### What Gets Removed from loop.py
- _snapshot_continuity_context() method (~50 lines)
- _clear_expired_continuity_context() method (~15 lines)
- continuity_context parameter threading through build_messages
- context_anchor metadata manipulation
- The massive inline compaction block (~120 lines) in _process_message
- _compaction_would_execute() preflight hack
- _plan_hybrid_batches() — replaced by cut point logic

### What Gets Simplified
- _consolidate_memory() becomes just memory extraction, not context management
- Token tracking stays but feeds into should_compact() instead of custom threshold logic

### Estimated LOC Change
- Remove: ~300 lines from loop.py
- Add: ~200 lines in session/compaction.py (Phases 2+3)
- Add: ~50 lines in session/manager.py (Phase 1)
- Modify: ~30 lines in context.py
- Net: cleaner, more modular, actually works

### Risk
- Compaction is the most critical path — if it breaks, agents lose context
- Need comprehensive tests before deploying
- Consider feature flag: config option to use old vs new compaction during transition

### Depends On
- compaction-entry (Phase 1)
- compaction-summary (Phase 2)
- compaction-cut-points (Phase 3)

---
schema_version: 1
id: compaction-cut-points
title: "Smart cut points \u2014 turn-aware compaction boundaries"
status: active
priority: 1
created_at: '2026-02-28T08:59:18Z'
updated_at: '2026-02-28T21:40:12Z'
---

## Tasks
- [x] compaction-cut-points.0 Implement find_valid_cut_points() — scan messages to find indices where it's safe to cut. Valid cut points are: before a user message, or before an assistant message that starts a new turn (not mid-tool-chain). Never cut between assistant tool_calls and their tool results.
- [x] compaction-cut-points.1 Implement find_cut_point() — given messages and keep_recent_tokens budget, walk backwards accumulating token estimates until budget is reached, then snap to nearest valid cut point. Returns CutPointResult with first_kept_index, turn_start_index (if split), is_split_turn flag. {deps=[compaction-cut-points.0]}
- [x] compaction-cut-points.2 Implement estimate_message_tokens() — estimate token count for a message dict. Use chars/3 for text (conservative), 1200 tokens for images, include tool_calls JSON length. Use actual usage data from assistant messages when available (input_tokens + cache_read + cache_creation). {deps=[compaction-cut-points.1]}
- [x] compaction-cut-points.3 Implement should_compact() — check if estimated context tokens exceed (context_window - reserve_tokens). Returns bool. Uses last assistant message usage data when available, falls back to estimation. {deps=[compaction-cut-points.2]}
- [x] compaction-cut-points.4 Handle split turns — when cut point falls mid-turn (e.g. between tool results in a multi-tool sequence), identify the turn start, split into prefix (to summarize) and suffix (to keep). Generate separate turn_prefix_summary for the prefix. {deps=[compaction-cut-points.3]}
- [x] compaction-cut-points.5 Add test: find_valid_cut_points() never returns index between tool_calls and tool results {deps=[compaction-cut-points.4]}
- [x] compaction-cut-points.6 Add test: find_cut_point() respects keep_recent_tokens budget and snaps to valid boundary {deps=[compaction-cut-points.5]}
- [x] compaction-cut-points.7 Add test: split turn detection works when cut falls between tool results in same turn {deps=[compaction-cut-points.6]}
- [x] compaction-cut-points.8 DEPENDENCY: compaction-summary must be complete before this phase {deps=[compaction-summary.10]}
- [x] compaction-cut-points.9 REORDER NOTE: implement estimate_message_tokens() (task .2) BEFORE find_cut_point() (task .1). find_cut_point depends on token estimation.
- [x] compaction-cut-points.10 Add persisted usage snapshot — store last known total_input_tokens and message_index in session metadata after each LLM call. This replaces the ephemeral _last_input_tokens dict and survives restarts.
- [x] compaction-cut-points.11 Add previous compaction boundary guard — find_valid_cut_points() must never return indices before the last CompactionEntry's first_kept_index. Prevents re-summarizing already-compacted history.
- [x] compaction-cut-points.12 Add no-op detection — if no summarizable prefix exists (all messages are after last compaction, and total tokens < threshold), return None from find_cut_point() instead of forcing a cut.
- [x] compaction-cut-points.13 Add test: cut point landing exactly on tool_result message snaps forward to next valid boundary
- [x] compaction-cut-points.14 Add test: cut point landing on assistant message with tool_calls snaps forward past the tool chain
- [x] compaction-cut-points.15 Add test: session with existing CompactionEntry — new cut point never goes before first_kept_index

## Notes
## Codex Audit Fixes Applied

### Fix: Task ordering
estimate_message_tokens() must be implemented before find_cut_point() since the cut point algorithm depends on token estimates. Added reorder note.

### Fix: No embedded usage data in session messages
Current session messages don't store per-turn usage data. The _last_input_tokens dict is ephemeral (lost on restart). Added task to persist usage snapshot in session metadata so token-based compaction decisions survive restarts.

### Fix: Previous compaction boundary
Without guarding against the previous CompactionEntry's boundary, repeated compactions could re-summarize already-compacted history. Added guard task.

### Fix: No-op detection
If all messages are recent (after last compaction) and under threshold, find_cut_point() should return None instead of forcing a meaningless cut.

### Fix: Additional edge case tests
Added tests for exact boundary conditions: cut landing on tool_result, on assistant with tool_calls, and respecting existing CompactionEntry.
## Design Context

Phase 3: Port Pi's smart cut point logic. This is the algorithm that decides WHERE to cut the conversation.

### The Problem
Our current approach just uses a fixed keep_count (25 messages from the end). This is naive because:
1. It can cut in the middle of a tool call chain (assistant says 'let me check' → tool_calls → tool results)
2. 25 messages might be too many or too few depending on message size
3. It doesn't account for actual token usage

### Pi's Algorithm
1. Walk backwards from the end of the message list
2. Accumulate estimated tokens for each message
3. When accumulated >= keep_recent_tokens (10% of context window = ~20k tokens), stop
4. Snap to the nearest VALID cut point at or before that position
5. A valid cut point is: before a user message that starts a new turn

### Turn Boundary Rules
A 'turn' in Pi is: user_message → [assistant_message → tool_calls → tool_results]* → final_assistant_message
You can NEVER cut inside the bracket sequence. Valid cuts are only at turn boundaries.

### Split Turn Handling
Sometimes the cut point falls inside a very long turn (e.g. the agent made 20 tool calls). In this case:
1. Find the turn start (the user message)
2. The cut falls somewhere in the tool chain
3. Messages before the cut = 'turn prefix' → summarized separately
4. Messages after the cut = 'turn suffix' → kept in context
5. Both summaries (history + turn prefix) are combined in the CompactionEntry

### Token Estimation
Pi uses chars/3 as a conservative estimate (overestimates tokens). For assistant messages with usage data, it uses the actual reported token count. We should do the same — we already track total_input from Anthropic's response.

### Configuration
- context_window_tokens: 200,000 (Claude)
- reserve_tokens: 16,384 (~8% of window, headroom for completion)
- keep_recent_tokens: 20,000 (10% of window, what to keep in context)

### Files to Create/Modify
- nanobot/session/compaction.py — cut point logic (same file as Phase 2)
- Tests: tests/test_compaction_cut_points.py (new)

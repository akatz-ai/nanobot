---
schema_version: 1
id: compaction-entry
title: "CompactionEntry \u2014 first-class session record with structured summary"
status: active
priority: 1
created_at: '2026-02-28T08:58:20Z'
updated_at: '2026-02-28T21:40:05Z'
---

## Tasks
- [x] compaction-entry.0 Add CompactionEntry dataclass to session/manager.py — fields: summary (str), first_kept_index (int), tokens_before (int), file_ops (dict with read_files and modified_files lists), previous_summary (str|None), timestamp
- [x] compaction-entry.1 Add append_compaction() method to Session — creates CompactionEntry, serializes to JSONL with _type=compaction, appends to session file {deps=[compaction-entry.0]}
- [x] compaction-entry.2 Add get_last_compaction() method to Session — walks messages/entries backwards to find most recent CompactionEntry {deps=[compaction-entry.1]}
- [x] compaction-entry.3 Update Session.get_history() — if CompactionEntry exists, inject summary as first message, skip all messages before first_kept_index, return summary + kept messages {deps=[compaction-entry.2]}
- [x] compaction-entry.4 Update SessionManager.load() — parse _type=compaction lines from JSONL back into CompactionEntry objects alongside regular messages {deps=[compaction-entry.3]}
- [x] compaction-entry.5 Add test: CompactionEntry round-trips through JSONL serialize/deserialize {deps=[compaction-entry.4]}
- [x] compaction-entry.6 Add test: get_history() with CompactionEntry returns summary + kept messages, not full history {deps=[compaction-entry.5]}
- [x] compaction-entry.7 DEPENDENCY: session-persistence must be complete — persistence model must support mixed entry types before CompactionEntry can be implemented {deps=[session-persistence.8]}
- [x] compaction-entry.8 Update append_compaction() — write directly to JSONL file with _type=compaction (bypass checkpoint() which only accepts chat roles). Also append to session.compactions list in memory.
- [x] compaction-entry.9 Add hard clamping in get_last_compaction() — if first_kept_index > len(messages), clamp to len(messages). If first_kept_index < 0, clamp to 0. Log warning on clamp.
- [x] compaction-entry.10 Resolve summary injection ownership: inject compaction summary in Session.get_history() ONLY, not in ContextBuilder.build_messages(). get_history() returns [summary_msg, ...kept_messages]. build_messages() just passes through whatever get_history() returns.

## Notes
## Codex Audit Fixes Applied

### Fix: get_history() leading message strip
Current get_history() strips leading non-user messages (manager.py:252-256). The injected compaction summary must be exempt from this stripping — it should be injected as a system message or handled specially so it's not dropped.

### Fix: Summary injection ownership
Codex found conflict: Phase 1 task .3 says inject in get_history(), Phase 4 task .2 says inject in build_messages(). RESOLVED: inject in get_history() only. build_messages() passes through.

### Fix: append_compaction() bypasses checkpoint()
checkpoint() rejects non user/assistant/tool roles. append_compaction() must write directly to JSONL with _type=compaction.

### Fix: Index clamping
first_kept_index must be clamped on load and on access. Invalid indices (from malformed lines being skipped, or session truncation) must not crash.
## Design Context

This is Phase 1 of the compaction rewrite, inspired by Pi Agent Rust's approach.

### Why
Our current compaction is broken. We use last_consolidated + context_anchor as two separate tracking mechanisms that get out of sync. Consolidation extracts facts to MEMORY.md but doesn't produce a session summary — so after compaction, the agent has no idea what it was just doing. The continuity_context hack (last 3-4 exchanges) is a band-aid.

### Pi's Approach
Pi stores compaction as a first-class entry in the session JSONL timeline:
```
[user] [assistant] [tool] [tool_result] ... [COMPACTION_ENTRY] [user] [assistant] ...
```

The CompactionEntry contains:
- summary: structured LLM-generated summary (Goal/Progress/Decisions/Next Steps/Critical Context)
- first_kept_entry_id: pointer to where live messages start
- tokens_before: token count before compaction
- details: file tracking (read_files, modified_files)

When building context: find last CompactionEntry → inject summary as system message → skip old messages → include kept messages.

### Key Difference from Pi
Pi uses entry IDs (UUIDs) for first_kept_entry_id. We use integer indices since our session is a flat list. We'll use first_kept_index (int) instead.

### Replaces
- context_anchor metadata field
- last_consolidated counter (partially — still needed for memory extraction tracking)
- continuity_context hack (Phase 4)

### Files to Modify
- nanobot/session/manager.py — CompactionEntry, append_compaction(), get_last_compaction(), get_history()
- Tests: tests/test_compaction_entry.py (new)

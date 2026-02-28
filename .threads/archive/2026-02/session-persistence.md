---
schema_version: 1
id: session-persistence
title: "Session persistence refactor \u2014 typed timeline model for mixed entry types"
status: active
priority: 1
created_at: '2026-02-28T09:16:08Z'
updated_at: '2026-02-28T21:40:03Z'
---

## Tasks
- [x] session-persistence.0 Design decision: keep session.messages as chat-only list, add separate session.compactions list for CompactionEntry objects. Serialization writes both; deserialization routes by _type field.
- [x] session-persistence.1 Update Session.save() — currently rewrites metadata + messages only (manager.py:453-464). Must also serialize compaction entries to JSONL with _type=compaction, interleaved at correct timeline position or appended after messages. {deps=[session-persistence.0]}
- [x] session-persistence.2 Update SessionManager._load() — parse _type=compaction lines into session.compactions list instead of skipping them. Preserve insertion order relative to messages for timeline reconstruction. {deps=[session-persistence.1]}
- [x] session-persistence.3 Update Session.clear() — must also clear session.compactions list, not just messages (manager.py:288-293). {deps=[session-persistence.2]}
- [x] session-persistence.4 Update validate_compaction_invariants() — replace last_consolidated <= context_anchor check with last_consolidated <= compaction.first_kept_index validation. Add clamping for stale/invalid first_kept_index on load. {deps=[session-persistence.3]}
- [x] session-persistence.5 Add backward compatibility: old JSONL files without _type=compaction lines must load cleanly with empty compactions list. New files with compaction entries must be readable by old code (graceful skip of unknown _type lines). {deps=[session-persistence.4]}
- [x] session-persistence.6 Add test: Session with compaction entries survives full save() rewrite cycle — save, reload, verify compaction entries preserved {deps=[session-persistence.5]}
- [x] session-persistence.7 Add test: Old JSONL files without compaction entries load cleanly {deps=[session-persistence.6]}
- [x] session-persistence.8 Add test: Malformed _type=compaction lines are skipped with warning, not crash {deps=[session-persistence.7]}

## Notes
## Design Context

Phase 0 — prerequisite for the entire compaction rewrite. Added based on Codex audit finding that the current Session persistence model silently drops non-message entries.

### The Problem (from Codex audit)
- Session.save() rewrites only metadata + session.messages (manager.py:453-464). Non-message timeline entries are dropped.
- Session.checkpoint() only accepts user|assistant|tool roles (manager.py:151-155), so CompactionEntry cannot reuse the checkpoint pipeline.
- If CompactionEntry is mixed into session.messages, downstream code that assumes chat records only will break (_consolidate_memory, resume detection, history sanitization).

### Solution
Keep session.messages as a pure chat message list. Add session.compactions as a separate list for CompactionEntry objects. Both are serialized to the same JSONL file (interleaved or appended), distinguished by _type field.

### Serialization Format
```jsonl
{"_type": "metadata", "key": "discord:123", ...}
{"role": "user", "content": "hello", ...}
{"role": "assistant", "content": "hi there", ...}
...
{"_type": "compaction", "summary": "## Goal\n...", "first_kept_index": 42, "tokens_before": 150000, ...}
{"role": "user", "content": "next question", ...}
...
```

### Files to Modify
- nanobot/session/manager.py — Session dataclass, save(), _load(), clear(), validate_compaction_invariants()
- Tests: tests/test_session_persistence.py (new)

### Depends On
Nothing — this is the foundation.

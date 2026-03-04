---
schema_version: 1
id: mem-phase0-bugfix
title: "Memory Phase 0 \u2014 Critical bug fixes (stop the bleeding)"
status: active
priority: 1
tags:
- memory,bugfix
created_at: '2026-03-04T08:15:50Z'
updated_at: '2026-03-04T08:17:15Z'
---

## Tasks
- [ ] mem-phase0-bugfix.0 Fix skipped-batch cursor advancement in hybrid compaction — cursor must not advance on failed extraction batches
- [ ] mem-phase0-bugfix.1 Fix session-scoped retrieval — hybrid_search/vector_search/keyword_search must use peer_key not agent_id when peer_key is provided
- [ ] mem-phase0-bugfix.2 Fix shared visibility — either implement shared retrieval semantics or remove shared from config/tool enums
- [ ] mem-phase0-bugfix.3 Preserve rich memory_type through hybrid path — stop coercing preference→fact, todo→goal in _to_history_entry()
- [ ] mem-phase0-bugfix.4 Add extraction/retrieval metrics — extraction lag (turns since last), skipped batches, retrieval hits by scope, prompt contribution by block
- [ ] mem-phase0-bugfix.5 Add tests for all Phase 0 fixes — cursor rollback on failure, session isolation, type preservation

## Notes
Oracle audit (2026-03-04) identified these as critical correctness issues in the live system. Cursor advancement bug can permanently lose memories. Session scoping leak means cross-session memory bleed. These must be fixed before any redesign work. Reference: agents/shared/oracle/nanobot-memory-redesign-audit-v6.md

---
schema_version: 1
id: mem-phase1-provenance
title: "Memory Phase 1 \u2014 Provenance foundation (stable IDs + extraction state)"
status: active
priority: 1
tags:
- memory,schema
created_at: '2026-03-04T08:16:14Z'
updated_at: '2026-03-04T08:17:15Z'
---

## Tasks
- [ ] mem-phase1-provenance.0 Assign stable message_id and message_ordinal during session checkpointing — deterministic IDs from JSONL order {deps=[mem-phase0-bugfix.5]}
- [ ] mem-phase1-provenance.1 Derive and persist turn_ordinal by counting user turns in session {deps=[mem-phase0-bugfix.5]}
- [ ] mem-phase1-provenance.2 Backfill legacy sessions on load — assign deterministic IDs to old sessions missing message_id/ordinals {deps=[mem-phase0-bugfix.5]}
- [ ] mem-phase1-provenance.3 Add SessionMemoryState to session metadata — last_extracted_message_ordinal, last_extracted_turn_ordinal, last_extraction_at, consecutive_failures, pending_batch_start_ordinal {deps=[mem-phase0-bugfix.5]}
- [ ] mem-phase1-provenance.4 Replace loose dict memory_graph config with typed MemoryConfig in nanobot config schema {deps=[mem-phase0-bugfix.5]}
- [ ] mem-phase1-provenance.5 Add tests — ordinal assignment, backfill idempotency, SessionMemoryState persistence across save/load {deps=[mem-phase0-bugfix.5]}

## Notes
Foundation for the graph+evidence-first redesign. Without stable message/turn IDs, provenance in MemoryEvidenceV2 would be meaningless. SessionMemoryState enables extraction to be decoupled from compaction with its own watermark. Typed config replaces the loose dict plumbing that causes drift between agent-memory and nanobot. Reference: agents/shared/oracle/nanobot-memory-redesign-audit-v6.md

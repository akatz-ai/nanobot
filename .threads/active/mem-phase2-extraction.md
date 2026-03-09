---
schema_version: 1
id: mem-phase2-extraction
title: "Memory Phase 2 \u2014 V2 schema + periodic graph-only extraction"
status: active
priority: 3
tags:
- memory,schema,extraction
created_at: '2026-03-04T08:16:15Z'
updated_at: '2026-03-09T06:36:43Z'
---

## Tasks
- [ ] mem-phase2-extraction.0 Create MemoryRecordV2 LanceDB model — id, owner_agent_id, scope (session|agent|shared|global), scope_id, record_type, status, pinned, content, normalized_content, fingerprint, vector, importance, confidence, decay_class (sticky|normal|ephemeral), entities, topics, source_kind, timestamps, occurrence_count, access_count, superseded_by {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.1 Create MemoryEvidenceV2 LanceDB model — memory_id, session_key, turn/message ordinals, message_ids, extraction_run_id, extractor_model, confidence, excerpt {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.2 Create MemoryEdgeV2 LanceDB model — source_id, target_id, relation_type (related_to|reinforces|supersedes|contradicts|about|caused_by), weight, provenance_evidence_id {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.3 Implement MemoryStoreV2 repository/service layer — CRUD for v2 records, evidence, edges with proper scope filtering {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.4 Implement dedup-as-reinforcement logic — on cosine match: bump occurrence_count + add evidence instead of overwriting content; on correction: create new record + supersedes edge {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.5 Implement SessionMemoryExtractor — extract from messages since last_extracted_message_ordinal, write to v2 graph + evidence only (not MEMORY.md) {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.6 Wire extraction triggers — every 12-15 completed user turns, on /new (session clear), on idle flush (30min), catch-up before compaction cut point if lagging {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.7 Extraction safety — extract only complete turns, batch by input-token budget, commit watermark only after successful batch, no cursor advancement on failure {deps=[mem-phase1-provenance.5]}
- [ ] mem-phase2-extraction.9 Add tests — extraction cadence, watermark advancement, failure rollback, dedup-vs-reinforce decisions, /new flush, idle flush trigger {deps=[mem-phase1-provenance.5]}

## Notes
Priority update (2026-03-09): periodic extraction decoupling is now deprioritized. Current compaction cadence appears frequent enough that running graph extraction more often is not worth the added complexity right now. Keep the correctness work (watermarking, no cursor drift, resumability) but defer independent extraction triggers unless compaction frequency or UX proves insufficient later.

Core of the redesign: new V2 schema (MemoryRecordV2 + MemoryEvidenceV2 + MemoryEdgeV2) and periodic graph-only extraction. Key design decisions: (1) dedup becomes reinforcement not overwrite, (2) decay classes (sticky/normal/ephemeral) replace flat importance, (3) extraction writes to graph only not MEMORY.md, (4) separate extraction watermark from compaction cursor. Schema design from Oracle audit Section 5. Reference: agents/shared/oracle/nanobot-memory-redesign-audit-v6.md

Simplified: no feature flag or shadow mode. V2 extraction replaces V1 directly. Just make sure tests pass and cut over. Back up LanceDB dir before migration if paranoid.

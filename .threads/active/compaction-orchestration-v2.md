---
schema_version: 1
id: compaction-orchestration-v2
title: "Compaction Orchestration V2 \u2014 explicit cut plans, persisted batch state,\
  \ and no-op-safe replay"
status: active
priority: 1
created_at: '2026-03-09T06:05:04Z'
updated_at: '2026-03-09T06:05:05Z'
---

## Tasks
- [ ] compaction-orchestration-v2.0 Persist explicit compaction plan metadata (summary range, extraction range, cut-point type) as first-class state
- [ ] compaction-orchestration-v2.1 Make compaction/extraction failure semantics explicit and resumable without cursor drift {deps=[compaction-orchestration-v2.0]}
- [ ] compaction-orchestration-v2.2 Add stable query/build helpers for compaction-visible history independent of Session.messages length {deps=[compaction-orchestration-v2.1]}

## Notes

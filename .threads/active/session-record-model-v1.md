---
schema_version: 1
id: session-record-model-v1
title: "Session Record Model V1 \u2014 explicit persisted session events and mutation\
  \ APIs"
status: active
priority: 2
created_at: '2026-03-09T06:05:04Z'
updated_at: '2026-03-09T06:07:00Z'
---

## Tasks
- [>] session-record-model-v1.0 Define typed persisted event model for chat rows, interruption markers, and compaction records {claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T06:06:59Z}
- [>] session-record-model-v1.1 Add explicit session mutation APIs and migrate remaining metadata/cursor updates off generic save() {deps=[session-record-model-v1.0] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T06:06:59Z}
- [>] session-record-model-v1.2 Add query helpers for prompt/replay construction from persisted records rather than Session object shape {deps=[session-record-model-v1.1] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T06:06:59Z}

## Notes

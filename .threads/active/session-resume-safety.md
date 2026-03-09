---
schema_version: 1
id: session-resume-safety
title: "Session Resume Safety \u2014 normalize interrupted tool calls and constrain\
  \ auto-resume"
status: active
priority: 1
created_at: '2026-03-09T06:03:28Z'
updated_at: '2026-03-09T06:05:04Z'
---

## Tasks
- [x] session-resume-safety.0 Normalize mid_tool sessions on startup by appending interrupted tool results and a clean assistant note
- [x] session-resume-safety.1 Restrict auto-resume scheduling to safe states after normalization {claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T06:03:41Z}
- [x] session-resume-safety.2 Add restart/resume regression tests for interrupted pending tool calls {deps=[session-resume-safety.1] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T06:03:41Z}
## Notes

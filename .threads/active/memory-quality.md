---
schema_version: 1
id: memory-quality
title: 'Memory quality: pinned/protected facts + supersedes semantics'
status: active
priority: 3
created_at: '2026-02-26T20:42:27Z'
updated_at: '2026-02-26T20:42:28Z'
---

## Tasks
- [ ] memory-quality.0 Add pinned/protected flag to MemoryRecord — prevents consolidation from dropping these
- [ ] memory-quality.1 Add superseded_by relation type to edge schema
- [ ] memory-quality.2 Add active boolean to preference/goal/decision memory types (latest wins + trail)
- [ ] memory-quality.3 Update consolidation to respect pinned memories and supersession chains
- [ ] memory-quality.4 Add dedup_threshold tuning: raise from 0.3 to 0.5+ to prevent aggressive overwrites

## Notes

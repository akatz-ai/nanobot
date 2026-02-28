---
schema_version: 1
id: compaction-cooldown
title: Compaction cooldown and hysteresis
status: active
priority: 2
created_at: '2026-02-26T20:41:47Z'
updated_at: '2026-02-28T09:01:20Z'
---

## Tasks
- [ ] compaction-cooldown.0 Add minimum turn count between compactions (e.g. 5 turns)
- [ ] compaction-cooldown.1 Add would-execute check: skip compaction if consolidation window is too small
- [ ] compaction-cooldown.2 Track last_compaction_turn in session metadata
- [ ] compaction-cooldown.3 Add test: rapid messages don't trigger back-to-back compactions

## Notes
## SUPERSEDED
This thread is superseded by the compaction rewrite epic:
- compaction-entry (Phase 1: CompactionEntry as session record)
- compaction-summary (Phase 2: Structured summary generation)
- compaction-cut-points (Phase 3: Smart turn-aware cut points)
- compaction-integration (Phase 4: Wire into agent loop)

The cooldown/hysteresis concerns are addressed by Phase 3's should_compact() which uses actual token usage data and reserve_tokens buffer instead of arbitrary cooldown timers.

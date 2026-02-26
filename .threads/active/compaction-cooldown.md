---
schema_version: 1
id: compaction-cooldown
title: Compaction cooldown and hysteresis
status: active
priority: 2
created_at: '2026-02-26T20:41:47Z'
updated_at: '2026-02-26T20:41:48Z'
---

## Tasks
- [ ] compaction-cooldown.0 Add minimum turn count between compactions (e.g. 5 turns)
- [ ] compaction-cooldown.1 Add would-execute check: skip compaction if consolidation window is too small
- [ ] compaction-cooldown.2 Track last_compaction_turn in session metadata
- [ ] compaction-cooldown.3 Add test: rapid messages don't trigger back-to-back compactions

## Notes

---
schema_version: 1
id: cross-agent-memory
title: Cross-agent memory retrieval (keep disabled by default)
status: active
priority: 3
created_at: '2026-02-26T20:42:00Z'
updated_at: '2026-02-26T20:42:01Z'
---

## Tasks
- [ ] cross-agent-memory.0 Add cross_agent config block to memory_graph config
- [ ] cross-agent-memory.1 Implement agent_ids filter + include_shared/global/include_other_private in store.recall()
- [ ] cross-agent-memory.2 Add cross-agent params to memory_recall and memory_graph tools
- [ ] cross-agent-memory.3 Add feature flag: cross_agent.enabled (default false)
- [ ] cross-agent-memory.4 Add test: agent A cannot see agent B private memories; can see shared/global

## Notes

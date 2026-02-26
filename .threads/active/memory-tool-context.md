---
schema_version: 1
id: memory-tool-context
title: "Memory tool set_context \u2014 wire agent_id through tool stack"
status: active
priority: 2
created_at: '2026-02-26T20:41:53Z'
updated_at: '2026-02-26T20:42:39Z'
---

## Tasks
- [ ] memory-tool-context.0 Add set_context(agent_id, peer_key) to memory tools in agent_memory_nanobot/tools.py
- [ ] memory-tool-context.1 Update AgentLoop._set_tool_context() to pass agent_id and session_key to memory tools
- [ ] memory-tool-context.2 Default memory_recall to agent_id scope instead of session_key scope
- [ ] memory-tool-context.3 Add test: memory tools receive correct agent_id context
- [ ] memory-tool-context.4 Wire agent_id from memory-tenancy into tool context defaults {deps=[memory-tenancy.5]}

## Notes

---
schema_version: 1
id: memory-tenancy
title: "Memory graph tenancy \u2014 agent_id as partition key, fix cross-session recall"
status: done
priority: 1
created_at: '2026-02-26T20:41:41Z'
updated_at: '2026-02-26T21:14:23Z'
---

## Tasks
- [x] memory-tenancy.0 Add agent_id and visibility columns to MemoryRecord and edge schema
- [x] memory-tenancy.1 Add _ensure_columns migration for agent_id + visibility in MemoryGraphStore
- [x] memory-tenancy.2 Update store.recall() to filter by agent_id (requester) instead of session-scoped peer_key
- [x] memory-tenancy.3 Update store.get_neighbors() to enforce agent_id filtering
- [x] memory-tenancy.4 Update hybrid.compact() to write agent_id on extracted memories
- [x] memory-tenancy.5 Update NanobotMemoryModule to propagate agent_id from workspace into all memory operations
- [x] memory-tenancy.6 Add test: memories saved in session A are retrievable in session B for the same agent

## Notes
Source: GPT-5.2 Pro audit finding #2. Graph memory currently uses peer_key=session_key, making it effectively session-local RAG instead of long-term cross-session memory. Switching to agent_id as the partition key aligns with the stated goal of persistent memory across conversations.

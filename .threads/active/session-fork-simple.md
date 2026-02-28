---
schema_version: 1
id: session-fork-simple
title: "Session forking (minimal) \u2014 clone-prefix forks using flat sessions"
status: active
priority: 3
created_at: '2026-02-28T09:18:31Z'
updated_at: '2026-02-28T09:18:52Z'
---

## Tasks
- [ ] session-fork-simple.0 Store source platform message IDs on persisted user messages — add optional message_id field to session message dicts during checkpoint(). Discord handler already passes message_id in metadata; persist it on the message itself. {deps=[compaction-integration.11]}
- [ ] session-fork-simple.1 Implement Session.fork_at_index(index) — copy messages[0:index] into a new Session object with a new session key. No entry_id/parent_id DAG needed. Returns new Session. {deps=[compaction-integration.11,session-fork-simple.0]}
- [ ] session-fork-simple.2 Wire Discord thread creation to fork — when a Discord thread is created off a message, use session_key_override mechanism (already exists in InboundMessage) to route thread messages to a forked session. Copy parent session history up to the thread's source message. {deps=[compaction-integration.11,session-fork-simple.1]}
- [ ] session-fork-simple.3 Implement /fork slash command (simple) — fork at message index N. Creates new session with copied prefix, responds with new session key. No tree navigation needed. {deps=[compaction-integration.11,session-fork-simple.2]}
- [ ] session-fork-simple.4 Add test: fork_at_index creates independent session with correct prefix {deps=[compaction-integration.11,session-fork-simple.3]}
- [ ] session-fork-simple.5 Add test: forked session compacts independently — CompactionEntry in fork doesn't affect parent {deps=[compaction-integration.11,session-fork-simple.4]}
- [ ] session-fork-simple.6 Add test: Discord thread creates forked session via session_key_override {deps=[compaction-integration.11,session-fork-simple.5]}

## Notes
## Design Context

Phase 5A — minimal, feasible session forking. Split from the original session-forking thread based on Codex audit recommendation.

### Why Split
Codex found that the original session-forking thread was 'massively under-scoped' — adding entry_id/parent_id DAG to the flat session architecture is a major rewrite. This 5A thread implements the useful parts without the architectural overhaul.

### Approach
Simple clone-prefix fork:
1. Copy messages[0:N] into a new Session
2. New session gets its own session key
3. Fork and parent are fully independent (no shared state)
4. Uses existing session_key_override in InboundMessage for Discord thread routing

### Key Insight from Codex
The session_key_override mechanism already exists in the bus model (events.py:19-24, base.py:93-124). Discord threads can use this to route to forked sessions without any DAG infrastructure.

### What This Doesn't Do
- No tree navigation (/tree command)
- No branch visualization
- No entry_id/parent_id relationships
- No merge or rebase between branches
Those are in session-forking (5B) if ever needed.

### Files to Modify
- nanobot/session/manager.py — fork_at_index(), message_id persistence
- nanobot/channels/discord.py — thread creation handler
- nanobot/agent/loop.py — /fork command
- Tests: tests/test_session_fork.py (new)

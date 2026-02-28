---
schema_version: 1
id: session-forking
title: "Session forking \u2014 conversation branching for Discord threads"
status: active
priority: 3
created_at: '2026-02-28T09:00:41Z'
updated_at: '2026-02-28T09:19:04Z'
---

## Tasks
- [ ] session-forking.0 Add entry_id field to session messages — UUID or monotonic counter, assigned on checkpoint(). Needed for fork point references.
- [ ] session-forking.1 Add parent_id field to session messages — points to previous message's entry_id. Enables tree structure. Linear sessions just chain sequentially. {deps=[session-forking.0]}
- [ ] session-forking.2 Implement Session.fork_from(entry_id) — creates a new Session containing all messages up to entry_id (the path from root to fork point). Returns new Session with shared history prefix. {deps=[session-forking.1]}
- [ ] session-forking.3 Map Discord threads to session forks — when a Discord thread is created off a message, look up the message in the parent channel's session, fork from that point, create a new session for the thread. {deps=[session-forking.2]}
- [ ] session-forking.4 Implement /fork slash command — fork the current session at a specified message (by index or content search). Creates a new session branch. {deps=[session-forking.3]}
- [ ] session-forking.5 Add test: fork creates new session with correct history prefix {deps=[session-forking.4]}
- [ ] session-forking.6 Add test: forked session can compact independently without affecting parent {deps=[session-forking.5]}
- [ ] session-forking.7 Add test: Discord thread mapping creates fork at correct message {deps=[session-forking.6]}
- [ ] session-forking.8 DEPENDENCY: compaction-integration must be complete before forking work begins {deps=[compaction-integration.11]}

## Notes
## RECLASSIFIED AS PHASE 5B

Based on Codex audit: this is a major architecture refactor, not a light feature. Requires:
- Typed entry DAG with entry_id/parent_id on every message
- Migration plan for all existing session files
- Dashboard parser updates
- Tree traversal and navigation commands

### Prerequisite
Phase 5A (session-fork-simple) should be implemented first as the minimal viable version. This thread (5B) is a separate project that should only be pursued if tree-structured sessions prove necessary.

### Codex Findings
- 'This phase assumes tree-structured session records can be added incrementally. In this codebase, that is a major rewrite, not a small feature.'
- 'Discord thread mapping is not implementable as written because current session messages do not store Discord source message IDs'
- 'High risk of turning P3 work into a long-running schema migration project'

Phase 5A addresses the Discord message ID storage and simple forking without the DAG.
## Design Context

Lower priority (P3). Conversation branching inspired by Pi Agent Rust's tree-structured sessions.

### Pi's Approach
Pi stores every session entry with an id and parent_id, forming a tree:
```
entry_0 (root) → entry_1 → entry_2 → entry_3 (branch A)
                                    ↘ entry_4 (branch B, forked from entry_2)
```

The /fork command:
1. User selects a message to fork from
2. Pi builds the path from root to that message's parent
3. Copies those entries into a new session
4. Pre-fills the editor with the original user message

### Nanobot Mapping
This maps naturally to Discord threads:
- Parent channel = main branch
- Discord thread off a message = fork from that point
- The thread gets all context up to the fork point
- Thread and channel can diverge independently
- Each branch can compact independently

### Implementation Considerations
- Entry IDs could be monotonic integers (simpler than UUIDs, sufficient for our use case)
- parent_id is optional — existing sessions without it are treated as linear chains
- Fork creates a NEW session file with copied messages, not a reference
- This means forked sessions are fully independent — no shared state bugs
- CompactionEntry in a forked session only covers that branch's history

### When This Becomes Useful
- Discord thread support (users ask questions in threads)
- A/B testing different approaches to a task
- 'What if' scenarios — fork, try something, discard if it doesn't work
- Multi-user conversations where each user branches off

### Depends On
- compaction-entry (Phase 1) — CompactionEntry must work with forked sessions
- compaction-integration (Phase 4) — forked sessions need independent compaction

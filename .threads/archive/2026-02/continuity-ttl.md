---
schema_version: 1
id: continuity-ttl
title: "Continuity context TTL \u2014 make post-compaction context ephemeral"
status: done
priority: 1
created_at: '2026-02-26T20:41:22Z'
updated_at: '2026-02-26T21:02:36Z'
---

## Tasks
- [x] continuity-ttl.0 Add continuity_expires_at_message_count to session metadata on compaction
- [x] continuity-ttl.1 Clear continuity_context when message count exceeds expiry threshold
- [x] continuity-ttl.2 Add test: continuity context is injected on first turn after compaction but cleared after N turns

## Notes
Source: GPT-5.2 Pro audit finding #4. Continuity context currently persists in session metadata and is injected every turn indefinitely, causing token bloat and stale guidance that no longer reflects the actual conversation state.

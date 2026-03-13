---
schema_version: 1
id: session-store-arch-v2
title: "Session Store Architecture V2 \u2014 DB-authoritative sessions and op-based\
  \ persistence"
status: active
priority: 1
created_at: '2026-03-09T05:31:21Z'
updated_at: '2026-03-09T06:05:04Z'
---

## Tasks
- [x] session-store-arch-v2.0 Stop stale cache write-back: metadata-only saves must not recompute message_count from unloaded LazyMessageList {claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:31:34Z}
- [x] session-store-arch-v2.1 Add DB-authoritative count repair on load and cache invalidation when session row count drifts {deps=[session-store-arch-v2.0] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:31:34Z}
- [x] session-store-arch-v2.2 Refactor prompt/history call sites to use DB-backed visible counts instead of trusting len(session.messages) as session authority {deps=[session-store-arch-v2.1] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:41:13Z}
- [x] session-store-arch-v2.3 Add optimistic session revision tracking for stale-cache detection across workers {deps=[session-store-arch-v2.2] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:41:13Z}
- [x] session-store-arch-v2.4 Add regression tests for stale flush, count mismatch repair, and Forge-style tail truncation {deps=[session-store-arch-v2.3] claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:31:34Z}
- [x] session-store-arch-v2.5 Introduce explicit save_state row-level persistence API and migrate metadata/cursor-only loop call sites {claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:44:53Z}
- [x] session-store-arch-v2.6 Detect loaded LazyMessageList content mutations in SQLite message signatures {claim_by=codex@/data/projects/nanobot claim_at=2026-03-09T05:44:53Z}

## Notes
Goal: move Nanobot away from write-back cached `Session` objects and toward an
op-based, DB-authoritative session model similar to opencode.

Planned migration shape:

1. Stabilize current SQLite store without changing higher-level APIs.
   - Clean `save()` / `save_all()` must no-op for unchanged sessions.
   - Metadata-only saves must preserve persisted row counts.
   - Loads must repair derived metadata (`message_count`) from canonical message rows.
   - Clean cached sessions should reload when the DB fingerprint changes.

2. Remove prompt-building dependence on cached `len(session.messages)` as
   session authority.
   - Introduce DB-backed visible-history/window helpers.
   - Use prompt-window counts derived from SQLite, not cache metadata.

3. Add optimistic revision tracking.
   - Session row gets a monotonic revision incremented by all message /
     compaction / metadata mutations.
   - Cached sessions compare revision before reuse.

4. Continue decomposing session state toward typed records.
   - Keep chat rows canonical.
   - Keep compactions/tool-prune metadata as explicit persisted records.
   - Avoid full-session rewrites except for migrations / repair tools.

Current turn delivered step 1 partially:
- metadata-only save no longer recomputes `message_count`
- clean cached sessions no longer rewrite on shutdown
- load repairs `message_count` drift from canonical message rows
- clean cached sessions reload when DB fingerprint changes
- regression tests cover stale flush, drift repair, and clean-cache reload

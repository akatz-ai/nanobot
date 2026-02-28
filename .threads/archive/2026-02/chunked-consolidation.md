---
schema_version: 1
id: chunked-consolidation
title: "Chunked consolidation \u2014 bounded window multi-pass extraction"
status: done
priority: 1
created_at: '2026-02-26T20:41:34Z'
updated_at: '2026-02-26T21:07:03Z'
---

## Tasks
- [x] chunked-consolidation.0 Design chunking strategy: consolidate in batches of N messages or M tokens
- [x] chunked-consolidation.1 Implement incremental consolidation with extraction_run_id per batch
- [x] chunked-consolidation.2 Advance last_consolidated incrementally after each chunk (not all-at-once)
- [x] chunked-consolidation.3 Handle partial failure: if one chunk fails, preserve progress from completed chunks
- [x] chunked-consolidation.4 Add test: large session consolidates in multiple passes without data loss

## Notes
Source: GPT-5.2 Pro audit finding #1 (top priority). Current consolidation processes the entire backlog in one LLM call. This works until sessions get long enough that the consolidation call itself fails (context limit, latency, cost). Need bounded windows with incremental progress.

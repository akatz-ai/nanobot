---
schema_version: 1
id: background-model-config
title: "Background model config \u2014 use Haiku for compaction/consolidation/ingestion"
status: done
priority: 1
created_at: '2026-02-26T20:41:27Z'
updated_at: '2026-02-26T21:05:29Z'
---

## Tasks
- [x] background-model-config.0 Add background_model field to AgentsDefaults and AgentProfile in config schema
- [x] background-model-config.1 Wire background_model into compaction LLM calls (loop.py legacy consolidation)
- [x] background-model-config.2 Wire background_model into agent-memory-nanobot adapter as default for retrieval/consolidation/ingestion
- [x] background-model-config.3 Add config documentation for background_model

## Notes

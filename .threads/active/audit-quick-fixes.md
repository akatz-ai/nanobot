---
schema_version: 1
id: audit-quick-fixes
title: 'Quick audit fixes: web_search bug, embedding assertion, headroom budgeting'
status: done
priority: 1
created_at: '2026-02-26T20:41:16Z'
updated_at: '2026-02-26T20:58:56Z'
---

## Tasks
- [x] audit-quick-fixes.0 Fix web_search NameError: api_key → self.api_key in WebSearchTool.execute()
- [x] audit-quick-fixes.1 Add test for web_search tool execution (smoke test)
- [x] audit-quick-fixes.2 Add startup assertion: embedding_dim == schema_dim in MemoryGraphStore init
- [x] audit-quick-fixes.3 Fix memory headroom budgeting: use context_window - completion_headroom instead of max_tokens * 0.45

## Notes
Source: GPT-5.2 Pro architecture audit (Feb 2026). These are the lowest-effort, highest-impact fixes identified. The web_search bug is a straight NameError that breaks the tool at runtime.

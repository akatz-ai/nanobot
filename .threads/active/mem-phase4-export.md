---
schema_version: 1
id: mem-phase4-export
title: "Memory Phase 4 \u2014 Export, git serialization, and legacy cleanup"
status: active
priority: 3
tags:
- memory,export
created_at: '2026-03-04T08:16:42Z'
updated_at: '2026-03-04T08:41:16Z'
---

## Tasks
- [ ] mem-phase4-export.0 Implement deterministic markdown exporter — one file per memory with YAML frontmatter (Obsidian-compatible), manifest indexes, bounded evidence metadata {deps=[mem-phase3-retrieval.10]}
- [ ] mem-phase4-export.1 Implement importer from export format back into v2 tables — for disaster recovery and bootstrapping new instances {deps=[mem-phase3-retrieval.10]}
- [ ] mem-phase4-export.2 Optional git commit hook — auto-commit exports to memory/export/ directory {deps=[mem-phase3-retrieval.10]}
- [ ] mem-phase4-export.3 Retire hybrid auto-write behavior — remove HybridMemoryManager dual-write path, keep only graph writes {deps=[mem-phase3-retrieval.10]}
- [ ] mem-phase4-export.5 Add tests — export determinism, import round-trip, export-during-writes safety, git commit integration {deps=[mem-phase3-retrieval.10]}
- [ ] mem-phase4-export.6 Wire export to scheduler — periodic trigger via heartbeat or cron, on graceful exit, and manual operator trigger {deps=[mem-phase3-retrieval.10]}

## Notes
Final cleanup phase. Deterministic markdown export for version control and disaster recovery. Retire the hybrid dual-write path. This is lower priority and can wait until phases 0-3 are stable. Reference: agents/shared/oracle/nanobot-memory-redesign-audit-v6.md

Simplified: no bake period or compat flag removal. Just retire the dual-write path once Phase 3 is done. This is cleanup, not a careful migration.

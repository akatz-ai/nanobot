---
schema_version: 1
id: inbox-phase2
title: "Inbox system phase 2 \u2014 hardening and enhancements"
status: active
priority: 3
created_at: '2026-03-04T06:17:32Z'
updated_at: '2026-03-04T06:18:01Z'
---

## Tasks
- [ ] inbox-phase2.0 Cursor-based ack: move drain after checkpoint to prevent event loss on crash between drain and checkpoint
- [ ] inbox-phase2.1 Drain/append race: add file lock or use atomic write pattern to prevent silent event loss when append races with drain rename
- [ ] inbox-phase2.2 Event count cap: limit pending events per drain to prevent prompt size spikes from runaway recurring jobs
- [ ] inbox-phase2.3 /new legacy cleanup: verify /new clears recent_cron_actions from metadata before session.clear() (may already be handled)
- [ ] inbox-phase2.4 Configurable delivery content limits: 8KB clip per event is hardcoded, make configurable with optional full-content retrieval via tool call
- [ ] inbox-phase2.5 Heartbeat response filtering: filter or summarize low-value heartbeat responses before outbound Discord delivery
- [ ] inbox-phase2.6 Message tool cross-session bridging: route message tool sends to target session inbox for durable delivery
- [ ] inbox-phase2.7 Stream supersede/dedup: deduplicate or supersede status-spam events (e.g. repeated 'still working' from Codex monitoring)

## Notes
## Context

Phase 1 shipped in commits 4a8df30, 521b12a, 9471383. Core inbox system working in production.

### Codex Audit Findings (2026-03-03)
Codex reviewed the inbox implementation and found 5 issues:

1. **Drain/append race (task .1)** — drain() renames then reads while append() writes with no lock. Practical risk is low since cron callbacks are serialized per-agent and inbox writes only happen at callback end, drains at turn start. But a file lock would eliminate the theoretical window.

2. **Stale content recovery (FIXED in 9471383)** — _extract_cron_session_content could return content from previous cron firings. Fixed by scoping scan to current turn boundary.

3. **/new legacy cleanup (task .3)** — May already be handled since session.clear() wipes metadata. Needs verification.

4. **Drain→checkpoint crash window (task .0)** — Events removed before checkpoint. Cursor-based ack would make this durable.

5. **No event count cap (task .2)** — Per-event size limits exist (16KB/8KB) but no count limit. A backlog of many events can spike prompt size.

### Production validation
- Confirmed inbox drain working with full content recovery after suppress-reply fix
- 5 events drained successfully in comfygit-dev session (21:50:31 2026-03-03)
- 3 events drained successfully in nanobot-dev session with full Codex review content

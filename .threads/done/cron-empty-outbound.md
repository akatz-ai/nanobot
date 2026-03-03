---
schema_version: 1
id: cron-empty-outbound
title: "Cron empty outbound \u2014 blank message published when message tool already\
  \ sent to target"
status: done
priority: 3
created_at: '2026-03-01T20:23:57Z'
updated_at: '2026-03-01T20:23:59Z'
---

## Tasks
- [ ] cron-empty-outbound.0 Guard on_cron_job deliver path to skip publishing when response is None or empty string

## Notes
## Bug Report (discovered by Codex audit 2026-03-01)

**Root cause:** When message tool already sent to the target in a cron callback, _process_message suppresses the final reply (returns None). process_direct returns empty string. But on_cron_job still publishes 'response or ""' when deliver=True (commands.py:571-577), emitting empty outbound payloads.

**Evidence:**
- Suppress logic: loop.py:1329-1341
- process_direct returns empty: loop.py:1848-1849
- on_cron_job publishes anyway: commands.py:571-577

**Impact:** Low — empty messages sent to Discord (may show as blank or be silently dropped).

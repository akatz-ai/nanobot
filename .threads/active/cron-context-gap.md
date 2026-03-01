---
schema_version: 1
id: cron-context-gap
title: "Cron context gap \u2014 cron callback turns not visible in subsequent agent\
  \ context"
status: active
priority: 2
created_at: '2026-03-01T07:06:29Z'
updated_at: '2026-03-01T07:06:33Z'
---

## Tasks
- [ ] cron-context-gap.0 Investigate how cron callback turns are stored in session — are they persisted to JSONL? Are they in session.messages? Are they filtered out by get_history()?
- [ ] cron-context-gap.1 Ensure cron callback turns (cron system message + agent tool calls + agent response) appear in the visible history window for subsequent user turns
- [ ] cron-context-gap.2 Add test: after a cron callback turn, the next user turn's context includes the cron interaction in the message history

## Notes
## Bug Report

**Observed:** After a cron callback fired and the agent processed it (checked Codex status, reported completion, cleaned up monitoring job), the agent had NO memory of this in the next user turn. When the user asked a follow-up question, the agent said 'want me to check if Codex is still running?' — not knowing it had already reported completion moments ago.

**Root cause:** Cron callback turns are system-initiated messages that go through the agent loop and produce responses, but they are either:
- Not persisted to the session history in a way that's visible in subsequent turns, OR
- Persisted but excluded from the visible history window (e.g., filtered out as system messages)

The agent loses all awareness of what it did during cron callbacks. This causes:
1. Redundant status checks ('want me to check?') when it already checked
2. Contradictory behavior (reporting something is running when it already reported completion)
3. User confusion — the agent appears to have amnesia about its own recent actions

**Impact:** High — this affects ALL cron-based workflows (Codex monitoring, scheduled checks, reminders with follow-up actions). The agent cannot maintain continuity across cron-triggered and user-triggered turns.

**Potential fixes:**
1. Ensure cron callback turns (both the cron message and agent response) are persisted to session history as normal messages visible in subsequent context
2. If full persistence is too expensive, inject a brief summary system message: 'You previously handled cron job X: [1-line summary of what you did]' at the start of the next user turn
3. Add a session-level 'recent cron actions' metadata field that the agent can reference

**Related:** cron-truncation thread (the truncated response that preceded this gap)

**Session:** discord_1476048732343763189 (nanobot-dev), 2026-02-28 ~22:45-23:05 PST

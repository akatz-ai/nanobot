---
schema_version: 1
id: cron-truncation
title: "Cron callback truncation \u2014 agent response cut off by max_tokens before\
  \ completing report"
status: active
priority: 2
created_at: '2026-03-01T07:06:10Z'
updated_at: '2026-03-01T07:06:14Z'
---

## Tasks
- [ ] cron-truncation.0 Detect finish_reason=length on cron callback responses and auto-inject a continuation turn so the agent can complete its report
- [ ] cron-truncation.1 Add configurable max_tokens override for cron callbacks (default higher than normal turns, or uncapped)
- [ ] cron-truncation.2 Add tests: cron callback with large tool output triggers continuation; cron callback within budget completes normally

## Notes
## Bug Report

**Observed:** Agent's cron callback fired to report Codex completion. Agent said 'Let me clean up the monitoring job and report' then stopped — no actual report was delivered to the user.

**Root cause:** The cron callback runs through the same agent loop as a normal message. The flow was:
1. Cron fires → agent runs `tmux capture-pane` and `cat /tmp/codex-result-nanobot-dev.txt`
2. The Codex result file was large (detailed audit output)
3. Large tool result entered agent context
4. Agent started composing response but hit `max_tokens` on generation
5. Response truncated (`finish_reason=length`) and sent as-is — user gets incomplete output

**Impact:** User never receives the Codex results they were waiting for. Agent appears to have 'forgotten' mid-sentence.

**Potential fixes (in priority order):**
1. Detect `finish_reason=length` on cron callbacks and auto-continue — send a follow-up message to complete the response
2. Truncate/cap tool result content within cron callbacks (e.g., limit `cat` output to 4k chars, summarize the rest)
3. Increase `max_tokens` for cron callbacks specifically
4. Two-phase cron pattern: first phase detects completion + cleans up job; second phase reads and summarizes results as a separate turn

**Session:** discord_1476048732343763189 (nanobot-dev), 2026-02-28 ~22:40 PST
**Cron job:** 3455f4a5 (Check Codex session codex-nanobot-dev)

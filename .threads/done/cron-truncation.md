---
schema_version: 1
id: cron-truncation
title: "Cron callback truncation \u2014 agent response cut off by max_tokens before\
  \ completing report"
status: done
priority: 2
created_at: '2026-03-01T07:06:10Z'
updated_at: '2026-03-01T20:23:30Z'
---

## Tasks
- [ ] cron-truncation.0 Detect finish_reason=length on cron callback responses and auto-inject a continuation turn so the agent can complete its report
- [ ] cron-truncation.1 Add configurable max_tokens override for cron callbacks (default higher than normal turns, or uncapped)
- [ ] cron-truncation.2 Add tests: cron callback with large tool output triggers continuation; cron callback within budget completes normally

## Notes
## Codex Audit Update (2026-03-01)

**Verdict: PARTIALLY CONFIRMED** — The general analysis about shared max_tokens and missing finish_reason=length continuation is correct, but the specific incident root cause was WRONG.

### Corrected Root Cause
The actual incident was NOT finish_reason=length truncation. Codex found:
1. Usage log for cron:3455f4a5 shows final finish_reason='stop', not 'length'
2. A worker restart occurred mid-turn (after 'Let me clean up...')
3. Auto-resume derived routing from _split_session_key('cron:3455f4a5') → channel='cron', chat_id='3455f4a5'
4. Resumed session tried to deliver to discord:3455f4a5 (bogus channel ID) → Discord HTTP 400
5. The report WAS composed but delivered to the wrong place

### Evidence
- Usage log: /home/ubuntu/.nanobot/workspace/agents/nanobot-dev/sessions/cron_3455f4a5.usage.jsonl (shows 'stop' not 'length')
- Worker restart: gateway.log:45237-45238
- Resume misroute: gateway.log:45425-45426, 45451-45453
- finish_reason=length only handled for tool calls: loop.py:607-653
- Non-tool length responses sent as-is: loop.py:726-744

### Revised Fix Priority
1. Fix cron resume routing (see new thread: cron-resume-routing)
2. Add bounded continuation for non-tool finish_reason=length (still valuable as general hardening)
3. Tool output caps/summarization for cron callbacks (still valuable)
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

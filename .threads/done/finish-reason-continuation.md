---
schema_version: 1
id: finish-reason-continuation
title: "Agent continuation \u2014 auto-continue when finish_reason=length on non-tool\
  \ responses"
status: done
priority: 2
created_at: '2026-03-01T20:23:58Z'
updated_at: '2026-03-01T20:40:41Z'
---

## Tasks
- [ ] finish-reason-continuation.0 Add continuation logic in _run_agent_loop for finish_reason=length without tool calls — append partial content and re-prompt with 'Continue from where you left off', max 2 retries
- [ ] finish-reason-continuation.1 Add tests: finish_reason=length triggers continuation; continuation capped at max retries; normal stop not affected
- [ ] finish-reason-continuation.2 Tag synthetic continuation messages to distinguish from real user input (metadata flag or switch to system role)

## Notes
## Code Review Follow-up (2026-03-01)

**Issue: Synthetic continuation messages persisted as user messages**

The continuation logic injects `{"role": "user", "content": "Continue from where you left off."}` into the message list and checkpoints it to the session. These appear indistinguishable from real user messages in:
- Session JSONL files
- Dashboard conversation viewer
- History/context for future turns
- Compaction summaries

**Recommended fix:** Either:
1. Add a metadata flag to the synthetic message: `{"role": "user", "content": "Continue...", "synthetic": true}` — dashboard and compaction can filter/label these
2. Use a system message instead of user message: `{"role": "system", "content": "The previous response was truncated. Continue from where you left off."}` — more semantically correct, though some models respond differently to system vs user prompts

Option 2 is cleaner but needs testing to confirm models continue properly from a system prompt rather than user prompt.
## Feature/Bug (identified during cron-truncation audit 2026-03-01)

**Current behavior:** finish_reason=length is only handled when tool calls are present (loop.py:607-653). For non-tool responses, the truncated content is treated as final and sent as-is (loop.py:726-744).

**Desired behavior:** When finish_reason=length and no tool calls, auto-inject a continuation turn to let the agent complete its response. Cap at 1-2 continuation attempts to prevent infinite loops.

**Impact:** Any long response (cron reports, detailed analysis, code generation) can be silently truncated without the user knowing.

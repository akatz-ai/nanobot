---
schema_version: 1
id: cron-resume-routing
title: "Cron resume routing \u2014 resumed cron sessions deliver to wrong channel/chat"
status: done
priority: 1
created_at: '2026-03-01T20:23:57Z'
updated_at: '2026-03-01T20:24:01Z'
---

## Tasks
- [ ] cron-resume-routing.0 Store original channel and chat_id in cron session metadata when creating the session in on_cron_job (commands.py:565-570)
- [ ] cron-resume-routing.1 Update _resume_session or startup resume loop to read original channel/chat_id from session metadata for cron sessions, use that for tool context and delivery routing
- [ ] cron-resume-routing.2 Add test: cron session interrupted mid-turn resumes and delivers to original Discord channel, not to cron:<job_id>

## Notes
## Bug Report (discovered by Codex audit 2026-03-01)

**Observed:** After a worker restart mid-cron-callback, the auto-resume tried to deliver the response to discord:3455f4a5 (the job ID) instead of the original discord channel. Discord returned HTTP 400.

**Root cause:** _split_session_key('cron:3455f4a5') yields channel='cron', chat_id='3455f4a5'. This is used by _resume_session() as tool context (loop.py:762-770). MessageTool defaults to this context when no explicit target is given (message.py:86-88). The agent then calls message tool with channel='discord' but chat_id defaults to '3455f4a5' (the job ID, not a real Discord channel ID).

**The original target channel/chat_id is lost** — it's stored in the CronJob metadata (job.channel, job.chat_id) but not propagated to the session or available during resume.

**Evidence:**
- _split_session_key: loop.py:762-765
- _resume_session applies as tool context: loop.py:767-770
- MessageTool defaults to context chat_id: message.py:86-88
- Cron session file shows message tool sent to discord:3455f4a5: cron_3455f4a5.jsonl:10-11
- Gateway log confirms Discord 400: gateway.log:45451-45453

**Impact:** HIGH — Any cron callback that gets interrupted by a restart will fail to deliver its response. The work is done but the user never sees it.

**Related:** cron-truncation (this was the actual root cause of that incident)

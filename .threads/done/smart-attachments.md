---
schema_version: 1
id: smart-attachments
title: "Smart file attachment handling \u2014 save to workspace, guide agent to targeted\
  \ reads"
status: done
priority: 1
created_at: '2026-03-01T20:55:49Z'
updated_at: '2026-03-01T20:55:53Z'
---

## Tasks
- [ ] smart-attachments.0 In discord.py attachment handling, detect text-like files and compute metadata (file size, line count via wc -l or len(readlines)). For files over 10KB, change the hint format to include size/lines and a guidance message to use targeted reads
- [ ] smart-attachments.1 Add read_file soft cap: when read_file returns content over 30KB chars, append a warning to the result suggesting grep/head/tail for future reads of this file. Do NOT truncate — just advise.
- [ ] smart-attachments.2 Add tests: large text attachment gets structured hint with metadata; small text attachment keeps current behavior; image attachment unchanged; read_file over threshold includes advisory

## Notes
## Design

**Problem:** When users upload text files via Discord, the file is saved to ~/.nanobot/media/ and the agent's only awareness is a path string in the user message. The agent calls read_file on the full file, dumping the entire content into context (e.g. 85k chars = ~24k tokens for a log file). Both pruning and compaction protect recent turns, so this bloat persists.

**Solution:** Save non-image attachments to the agent's workspace uploads dir, inject a structured hint with file metadata (size, line count), and guide the agent to use targeted reads (grep, head, tail) instead of full read_file.

### Flow Change

**Before:**
1. Discord saves to ~/.nanobot/media/
2. User message gets `[attachment: /home/ubuntu/.nanobot/media/id_filename.txt]`
3. Agent calls read_file → full 85k dump into context
4. 24k tokens consumed, immune to pruning/compaction

**After:**
1. Discord saves non-image files to `{agent_workspace}/uploads/{filename}`
2. User message gets structured hint:
   `[File uploaded: filename.txt (85KB, ~1,200 lines) → uploads/filename.txt]`
   `Tip: use grep/head/tail to explore large files instead of reading the full content.`
3. Agent uses exec('grep -i error uploads/filename.txt') → ~700 tokens
4. 90% token reduction

### Implementation

**nanobot/channels/discord.py** — attachment handling (~line 360-379):
- Detect non-image files (text-like extensions: .txt, .log, .json, .py, .md, .csv, .yaml, .yml, .xml, .html, .sh, .cfg, .conf, .toml, .ini, .env)
- Instead of saving to ~/.nanobot/media/, save to agent workspace uploads dir
- Need: the channel handler needs to know the target agent's workspace path. This info flows through the message bus — check how _handle_message routes to agents and whether workspace path is available.
- Alternative: save to media dir as now, but ALSO copy/symlink to agent workspace. Simpler but duplicates files.

**Problem: channel doesn't know agent workspace at attachment time.**
The Discord channel handler processes attachments before routing to an agent. It doesn't know which agent will handle the message. Options:
1. Save to a shared uploads dir (e.g. ~/.nanobot/uploads/) and let the hint point there — simplest
2. Save to media dir as now, but change the hint format — no file move needed
3. Move the file in the agent loop after routing — two-phase approach

**Recommended: Option 2 (change hint only) + soft read_file guidance.**
Keep saving to ~/.nanobot/media/ (already works). Change the hint in discord.py to include file metadata. The agent loop can optionally copy/symlink to workspace on first access. This is the smallest change with the biggest impact.

### Hint Format
For text files over a size threshold (e.g. 10KB):
```
[File: filename.txt (85KB, ~1,200 lines) saved to /home/ubuntu/.nanobot/media/id_filename.txt]
This is a large file. Use grep, head, tail, or wc to explore it rather than reading the full content.
```

For small text files (under 10KB), keep current behavior — read_file is fine.

For binary/unknown files, keep current behavior.

### Token Savings
- Current: ~24,000 tokens for an 85KB log file
- Proposed: ~700-2,400 tokens (grep + targeted reads)
- Reduction: ~90%

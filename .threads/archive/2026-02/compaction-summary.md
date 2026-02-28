---
schema_version: 1
id: compaction-summary
title: "Structured compaction summary \u2014 Pi-style Goal/Progress/Decisions/Next\
  \ Steps prompt"
status: active
priority: 1
created_at: '2026-02-28T08:58:46Z'
updated_at: '2026-02-28T21:40:13Z'
---

## Tasks
- [x] compaction-summary.0 Define SUMMARIZATION_SYSTEM_PROMPT constant — 'You are a context summarization assistant. Your task is to read a conversation and produce a structured summary. Do NOT continue the conversation. ONLY output the structured summary.'
- [x] compaction-summary.1 Define SUMMARIZATION_PROMPT constant — structured format: Goal, Constraints & Preferences, Progress (Done/In Progress/Blocked), Key Decisions, Next Steps, Critical Context. Instruction to preserve exact file paths, function names, error messages. {deps=[compaction-summary.0]}
- [x] compaction-summary.2 Define UPDATE_SUMMARIZATION_PROMPT constant — for iterative updates when a previous summary exists. Rules: preserve existing info, add new progress, move In Progress to Done when completed, update Next Steps. {deps=[compaction-summary.1]}
- [x] compaction-summary.3 Define TURN_PREFIX_SUMMARIZATION_PROMPT constant — for split-turn scenarios where the cut point falls mid-turn. Summarizes the prefix to provide context for the kept suffix. {deps=[compaction-summary.2]}
- [x] compaction-summary.4 Implement generate_compaction_summary() async function — takes messages to summarize, provider, model, optional previous_summary. Serializes conversation into tagged XML block, appends appropriate prompt (new or update), calls LLM, returns structured summary text. {deps=[compaction-summary.3]}
- [x] compaction-summary.5 Implement generate_turn_prefix_summary() async function — for split-turn case, summarizes the early part of a turn that was cut. {deps=[compaction-summary.4]}
- [x] compaction-summary.6 Implement serialize_conversation() helper — converts message dicts to readable transcript format: [User]: text, [Assistant]: text, [Tool:name]: result. Handles tool_calls and tool results. {deps=[compaction-summary.5]}
- [x] compaction-summary.7 Add file operation tracking — extract_file_ops() walks messages, finds read/write/edit/bash tool calls, tracks which files were read vs modified. Only counts successful tool calls. {deps=[compaction-summary.6]}
- [x] compaction-summary.8 Add test: generate_compaction_summary() produces all required sections {deps=[compaction-summary.7]}
- [x] compaction-summary.9 Add test: generate_compaction_summary() with previous_summary uses update prompt and preserves existing info {deps=[compaction-summary.8]}
- [x] compaction-summary.10 Add test: file operation tracking correctly categorizes read vs write vs edit {deps=[compaction-summary.9]}
- [x] compaction-summary.11 DEPENDENCY: compaction-entry must be complete before this phase {deps=[compaction-entry.6]}
- [x] compaction-summary.12 REORDER NOTE: implement serialize_conversation() (task .6) BEFORE generate_compaction_summary() (task .4). The summary generator depends on the serializer, not the other way around.
- [x] compaction-summary.13 Fix tool name assumptions in extract_file_ops() — actual tool names are read_file, write_file, edit_file, list_dir, exec (not read, write, edit as in Pi). Also handle exec tool calls that write files via heredoc/redirect.
- [x] compaction-summary.14 Handle assistant messages with content=None and only tool_calls — serialize as [Assistant]: (tool calls only) with tool call names listed
- [x] compaction-summary.15 Handle non-string user content (image blocks, multi-part content) — extract text parts, note images as [image attached]
- [x] compaction-summary.16 Add transcript size control — cap serialized conversation at max_transcript_chars (default 100k chars) before sending to LLM. Truncate from the beginning (oldest messages) if over budget.
- [x] compaction-summary.17 Add output validation — verify generated summary contains required sections (Goal, Progress, Next Steps). If malformed, retry once. If still malformed, use a fallback template with raw conversation excerpt.

## Notes
## Codex Audit Fixes Applied

### Fix: Task ordering
Codex found that generate_compaction_summary() (task .4) depends on serialize_conversation() (task .6) and extract_file_ops() (task .7), but those were scheduled later. Added reorder note (task .12). Implementation order should be: .6 (serialize) -> .7 (file ops) -> .0-.3 (prompts) -> .4-.5 (generators) -> .8-.10 (tests).

### Fix: Tool name mismatch
Pi uses tool names read/write/edit. Nanobot uses read_file/write_file/edit_file/list_dir/exec. extract_file_ops() must use correct names. Also need to handle exec tool calls that create/modify files via shell commands.

### Fix: Missing content handling
- Assistant messages can have content=None with only tool_calls
- User content can be list of blocks (text + images), not just string
- Both must be handled in serialize_conversation()

### Fix: Unbounded transcript
Without size controls, a 200k-token conversation would produce a massive transcript that exceeds the summarization model's context window. Added max_transcript_chars cap.

### Fix: Output validation
LLM output is non-deterministic. Summary might be missing required sections or be malformed. Added validation + retry + fallback template.
## Design Context

This is Phase 2 of the compaction rewrite. Implements the LLM-powered summary generation that produces the content for CompactionEntry.

### Pi's Summary Format
Pi uses this exact structured format, which is excellent for agent continuity:

```
## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints mentioned by user]

## Progress
### Done
- [x] [Completed tasks/changes]
### In Progress
- [ ] [Current work]
### Blocked
- [Issues preventing progress]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, or references needed to continue]
```

### Iterative Summaries
When compacting again (previous CompactionEntry exists), Pi uses UPDATE_SUMMARIZATION_PROMPT which says: 'preserve all existing information, ADD new progress, move items from In Progress to Done when completed.' This means summaries accumulate knowledge instead of losing it.

### File Operation Tracking
Pi tracks which files were read vs modified across the compacted window. This is injected into the summary as XML tags:
```xml
<read-files>
src/main.rs
src/lib.rs
</read-files>
<modified-files>
src/agent.rs
</modified-files>
```

### Implementation Notes
- Use the background_model (typically Haiku) for summary generation, not the primary model
- Temperature 0.8 for summaries (Pi's setting — slightly creative but structured)
- Summary should be bounded to ~reserve_tokens (16k) to avoid the summary itself being too large
- Conversation serialization should handle our message format: role/content/tool_calls/tool_call_id

### Depends On
- compaction-entry (Phase 1) — CompactionEntry type must exist

### Files to Create/Modify
- nanobot/session/compaction.py (new) — summary generation, prompts, file tracking
- Tests: tests/test_compaction_summary.py (new)

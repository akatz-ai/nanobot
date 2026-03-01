---
schema_version: 1
id: batch-tool
title: "Batch tool \u2014 programmatic tool calling for multi-tool workflows"
status: active
priority: 2
created_at: '2026-03-01T06:21:05Z'
updated_at: '2026-03-01T06:21:45Z'
---

## Tasks
- [ ] batch-tool.0 Implement BatchTool class (batch.py): restricted namespace, async tool wrappers, stdout capture, timeout enforcement, guarded __import__
- [ ] batch-tool.1 Build tool wrapper generator: create async functions for each registered tool (read_file, write_file, edit_file, list_dir, exec_command, web_search, web_fetch) from ToolRegistry
- [ ] batch-tool.2 Add per-call logging inside batch execution (INFO level for each tool invocation, summary on completion) {deps=[batch-tool.0]}
- [ ] batch-tool.3 Register BatchTool in loop.py _register_default_tools() — pass ToolRegistry reference {deps=[batch-tool.1]}
- [ ] batch-tool.4 Write tests (test_batch_tool.py): ~15 cases covering basic execution, tool wrappers, multi-call, stdout capture, timeout, import restrictions, no raw file access, error handling, empty/truncated output, exec_command rename, async/await, JSON processing, realistic workflow {deps=[batch-tool.1]}
- [ ] batch-tool.5 Validate all existing tests still pass with batch tool registered {deps=[batch-tool.3]}

## Notes
## Architecture

**Concept:** A `batch` tool that lets the agent write Python scripts calling other nanobot tools as async functions. Tool results stay inside the script — only print() output enters conversation context.

**Execution model:** In-process via exec() with restricted namespace. Tool functions are async wrappers calling ToolRegistry.execute(). Stdout captured via StringIO.

**Exposed tools:** read_file, write_file, edit_file, list_dir, exec_command (renamed from exec), web_search, web_fetch. NOT exposed: message, spawn, cron, memory_* (side-effect-heavy, need conversation visibility).

**Safety:** Guarded __import__ (only json/re/math/datetime/collections/etc), no os/subprocess/sys/socket. Timeout via asyncio.wait_for() (default 120s, max 300s). All tool safety guards apply identically.

**Token savings:** comfygit-dev audit showed 8 web_fetch calls = ~400k chars in context. With batch: one tool call, ~16k chars.

**Full scope doc:** agents/nanobot-dev/programmatic-tool-calling-scope.md

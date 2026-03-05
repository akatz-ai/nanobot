# Memory Extraction v2 — Spec

## Overview

Redesign the memory extraction pipeline to decouple it from compaction, add session-local tool result indexing, and auto-compact MEMORY.md. This spec covers three interconnected changes that share a single trigger mechanism.

**Reference:** Oracle GPT-5.2 Pro audit (2026-03-04) — full audit at `/home/ubuntu/.nanobot/media/1479245243747401810_message.txt`

---

## Architecture

### Current State

```
Compaction fires (token pressure) →
  1. Generate structured summary (CompactionEntry)
  2. Extract memories from compacted range → graph + MEMORY.md
  3. Advance last_consolidated cursor
```

Problems:
- Extraction is coupled to compaction — only runs when token pressure triggers it
- If extraction fails, cursor still advances (silent memory loss)
- MEMORY.md grows unbounded (nanobot-dev is 20KB, should be ~5K)
- Tool results vanish after pruning/compaction with no way to search back
- Extraction writes shared facts to MEMORY.md (agent-private file)

### Target State

```
Every ~20 user turns OR on compaction OR on /new:
  1. Extract shared facts/decisions → memory graph (shared visibility)
  2. Index tool results → session evidence SQLite (session-local)
  3. Compact MEMORY.md if oversized → distill to agent-private preferences

Compaction fires independently (token pressure) →
  1. Generate structured summary (CompactionEntry) — unchanged
  2. Trigger catch-up extraction if extraction watermark lags
```

### Context Hierarchy (always-in-context → retrieved-per-turn)

```
Always in context (static, cached):
  SOUL.md         — shared personality (all agents)
  USER.md         — user profile (all agents)
  IDENTITY.md     — agent role/personality (per-agent)
  MEMORY.md       — agent private working notes (per-agent, capped ~5K tokens)

Retrieved per-turn (dynamic):
  Memory graph    — shared facts, decisions, preferences (from extraction)
  Session index   — tool results from this conversation (evidence SQLite)
  Knowledge base  — reference docs, specs, research (Obsidian vault)
```

---

## Component 1: Periodic Extraction (Decoupled from Compaction)

### Trigger Conditions

Extraction runs when ANY of:
1. **Turn count:** 20 completed user turns since last extraction
2. **Compaction catch-up:** Compaction is about to cut messages that haven't been extracted yet
3. **Session end:** `/new` command or session archive
4. **Idle flush:** 30 minutes since last user message (via heartbeat)

### Extraction Watermark

Add to `Session.metadata`:

```python
class ExtractionState:
    last_extracted_index: int = 0        # message index through which extraction completed
    last_extraction_at: str | None       # ISO timestamp
    consecutive_failures: int = 0        # reset on success, increment on failure
    pending_batch_start: int | None      # non-None while a batch is in-flight
```

**Critical rule:** The watermark advances ONLY after successful extraction commit. Never on failure. This fixes the silent memory loss bug from the Oracle audit (Finding #1).

### Extraction Window

```python
start = session.metadata.extraction_state.last_extracted_index
end = len(session.messages)  # or compaction cut point if catch-up

# Give ~5 messages of overlap for context continuity
context_start = max(0, start - 5)
window = session.messages[context_start:end]
```

The extraction window is the messages between `last_extracted_index` and current end. We include 5 messages of backward overlap so the extractor has context continuity across runs.

### Extraction Model

Use `background_model` (currently Haiku 3.5). This is the same model used today — cheap and fast.

### Extraction Output

The extractor produces structured items, each categorized:

```python
class ExtractedItem:
    content: str
    memory_type: str          # fact | decision | preference | goal | event | procedure
    importance: float         # 0.0 - 1.0
    entities: list[str]
    scope: str                # "agent" | "shared"
    decay_class: str          # "sticky" | "normal" | "ephemeral"
```

**Routing rules:**
- `scope="shared"` → save to memory graph with shared visibility (all agents can retrieve)
- `scope="agent"` → save to memory graph with agent visibility (only this agent retrieves)
- Preferences about the agent's own behavior → don't extract, leave in MEMORY.md

### Extraction Prompt

The extraction prompt should explicitly instruct the model to:

```
You are extracting durable memories from a conversation transcript.

Categorize each extracted item:
- SHARED facts/decisions that any agent should know (e.g., "The API uses OAuth2",
  "Alex prefers dark mode", "We chose Fly.io for hosting")
- AGENT-SPECIFIC working context only relevant to this agent's current task
  (e.g., "Currently debugging the auth flow", "Need to follow up on PR #42")

Do NOT extract:
- Transient chatter or acknowledgments
- Information that's only relevant within this conversation (use session search for that)
- Agent self-preferences (those belong in MEMORY.md, not extracted here)

For each item, output:
- content: the fact/decision/preference in clear, standalone language
- type: fact | decision | preference | goal | event | procedure
- importance: 0.0-1.0
- entities: key named entities
- scope: "shared" or "agent"
- decay: "sticky" (permanent), "normal" (weeks), or "ephemeral" (days)
```

### Integration with Compaction

When compaction fires:

```python
# Before compaction cuts messages, check if extraction is behind
extraction_state = session.metadata.get("extraction_state", {})
last_extracted = extraction_state.get("last_extracted_index", 0)
cut_point = compaction_entry.first_kept_index

if last_extracted < cut_point:
    # Catch-up extraction for messages about to leave the window
    await extract_memories(session, start=last_extracted, end=cut_point)

# Then proceed with compaction as normal
```

This ensures we never compact away messages without extracting from them first.

### Failure Handling

```python
try:
    items = await run_extraction(session, start, end)
    save_to_graph(items)
    session.metadata["extraction_state"]["last_extracted_index"] = end
    session.metadata["extraction_state"]["consecutive_failures"] = 0
except Exception:
    session.metadata["extraction_state"]["consecutive_failures"] += 1
    # Do NOT advance watermark
    # After 3 consecutive failures, log warning and skip until next trigger
```

---

## Component 2: Session Evidence Index

### Purpose

Index tool results and significant outputs into a per-session SQLite FTS5 database so the agent can search back for specific implementations, error messages, file contents, etc. — even after they've been pruned from the prompt window or compacted away.

### Storage

Sidecar file alongside the session JSONL:
```
sessions/
  discord_123456.jsonl              # session messages
  discord_123456.compaction.jsonl   # compaction entries
  discord_123456.context.jsonl      # context log
  discord_123456.usage.jsonl        # usage log
  discord_123456.evidence.sqlite    # NEW: tool result index
```

### Schema

```sql
CREATE TABLE evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_index INTEGER NOT NULL,       -- global message index in session
    tool_name TEXT NOT NULL,               -- e.g., "exec", "read_file", "web_fetch"
    chunk_index INTEGER NOT NULL DEFAULT 0, -- for multi-chunk results
    content TEXT NOT NULL,                 -- the actual tool output chunk
    timestamp TEXT,                        -- ISO timestamp from message
    char_count INTEGER NOT NULL,           -- original content length
    metadata TEXT                          -- JSON: tool args summary, exit code, etc.
);

CREATE VIRTUAL TABLE evidence_fts USING fts5(
    content,
    tool_name,
    content='evidence',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER evidence_ai AFTER INSERT ON evidence BEGIN
    INSERT INTO evidence_fts(rowid, content, tool_name)
    VALUES (new.id, new.content, new.tool_name);
END;

CREATE TRIGGER evidence_ad AFTER DELETE ON evidence BEGIN
    INSERT INTO evidence_fts(evidence_fts, rowid, content, tool_name)
    VALUES ('delete', old.id, old.content, old.tool_name);
END;
```

### Indexing Rules

**When to index:** After every tool result is added to the session (hook in the agent loop post-tool-execution).

**What to index:**

| Tool | Index? | Notes |
|------|--------|-------|
| `exec` | ✅ Yes | Command outputs, build results, grep results |
| `read_file` | ✅ Yes | File contents — high value for searching back |
| `write_file` | ✅ Yes | What was written (the content parameter) |
| `edit_file` | ✅ Yes | The old_text/new_text for searching diffs |
| `web_fetch` | ✅ Yes | Fetched page content |
| `web_search` | ✅ Yes | Search results |
| `list_dir` | ✅ Yes | Directory listings |
| `memory_recall` | ❌ No | Already in the memory system |
| `memory_save` | ❌ No | Already in the memory system |
| `memory_graph` | ❌ No | Already in the memory system |
| `cron` | ❌ No | Small, stays in context |
| `message` | ❌ No | Cross-channel, not searchable content |
| `spawn` | ❌ No | Subagent delegation, not content |

**Minimum size threshold:** Only index results ≥ 512 characters. Small acknowledgments ("file written successfully") aren't worth indexing.

**Chunking:** Split large results into ~1200 character chunks with 100 character overlap for better FTS matching. Each chunk gets its own row with incrementing `chunk_index`.

### Indexing Hook

In `loop.py`, after tool execution completes and the result is added to the session:

```python
# After tool result is appended to session.messages
tool_msg = session.messages[-1]  # the tool result message
if tool_msg.get("role") == "tool":
    tool_name = tool_msg.get("name", "")
    content = tool_msg.get("content", "")
    if _should_index_tool(tool_name, content):
        evidence_index = self._get_evidence_index(session)
        evidence_index.add(
            message_index=len(session.messages) - 1,
            tool_name=tool_name,
            content=content,
            timestamp=tool_msg.get("timestamp"),
            metadata=_extract_tool_metadata(tool_msg),
        )
```

Also index tool results inside `batch.py` for the programmatic tool calling path.

### Search Tool

New tool exposed to the agent:

```python
class SessionSearchTool(Tool):
    name = "session_search"
    description = """Search this conversation's tool outputs and code.
    Use when you need to find something from earlier in this session — a file you read,
    a command you ran, an error message, code you wrote, or a web page you fetched.
    Returns matching excerpts with their original message index and tool name."""

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — use keywords from the content you're looking for"
            },
            "tool_filter": {
                "type": "string",
                "description": "Optional: filter by tool name (e.g., 'exec', 'read_file')"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
```

**Search implementation:**
```python
def search(self, query: str, tool_filter: str | None = None, max_results: int = 5) -> list[dict]:
    sql = """
        SELECT e.message_index, e.tool_name, e.timestamp, e.char_count,
               snippet(evidence_fts, 0, '>>>', '<<<', '...', 40) as excerpt,
               rank
        FROM evidence_fts
        JOIN evidence e ON evidence_fts.rowid = e.id
        WHERE evidence_fts MATCH ?
    """
    params = [query]
    if tool_filter:
        sql += " AND e.tool_name = ?"
        params.append(tool_filter)
    sql += " ORDER BY rank LIMIT ?"
    params.append(max_results)
    return self._execute(sql, params)
```

**Return format to the agent:**
```
Found 3 results:

[1] exec (message #1847, Mar 5 11:23)
  ...the OAuth token >>>refresh_token<<< endpoint returns a 401 when...
  (2,340 chars total)

[2] read_file (message #1632, Mar 4 16:05)
  ...def >>>refresh_token<<<(self, token: str) -> TokenResponse:...
  (4,120 chars total)

[3] web_fetch (message #1590, Mar 4 14:30)
  ...RFC 6749 Section 6: >>>Refreshing<<< an Access >>>Token<<<...
  (8,900 chars total)
```

### Seeding Existing Sessions

One-time migration script to backfill evidence for existing sessions:

```python
async def seed_evidence_index(session_path: Path):
    """Walk a session JSONL and index all tool results."""
    jsonl_path = session_path
    sqlite_path = session_path.with_suffix(".evidence.sqlite")

    if sqlite_path.exists():
        return  # already seeded

    index = EvidenceIndex(sqlite_path)
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f):
            msg = json.loads(line.strip())
            if msg.get("role") != "tool":
                continue
            if msg.get("_type") in ("metadata", "compaction"):
                continue
            tool_name = msg.get("name", "")
            content = msg.get("content", "")
            if _should_index_tool(tool_name, content):
                index.add(
                    message_index=line_num,
                    tool_name=tool_name,
                    content=content,
                    timestamp=msg.get("timestamp"),
                )
    index.close()
```

Run this on startup for sessions that have a JSONL but no `.evidence.sqlite`.

---

## Component 3: MEMORY.md Auto-Compaction

### Purpose

Keep MEMORY.md small and focused on agent-private working notes. Shared facts get extracted to the graph; the file itself gets periodically distilled.

### Trigger

During extraction runs (Component 1), after extraction completes:

```python
memory_md_path = agent_workspace / "memory" / "MEMORY.md"
content = memory_md_path.read_text()
token_count = estimate_tokens(content)

if token_count > MEMORY_COMPACT_THRESHOLD:  # 10,000 tokens
    compacted = await compact_memory_md(content, target_tokens=MEMORY_COMPACT_TARGET)  # 5,000 tokens
    memory_md_path.write_text(compacted)
```

### Compaction Prompt

Use the same `background_model` (Haiku 3.5):

```
You are compacting an agent's personal memory file. This file contains the agent's
private working notes — preferences, active project context, and operational knowledge.

Current file is {token_count} tokens. Distill it to ~{target_tokens} tokens.

Rules:
1. KEEP: Agent-specific preferences, working style notes, active task context
2. KEEP: Information that changes how this specific agent behaves
3. REMOVE: Shared facts about the project/user that other agents also know
   (these are already saved to the shared memory graph)
4. REMOVE: Stale/completed task context that's no longer relevant
5. REMOVE: Redundant entries that say the same thing multiple ways
6. MERGE: Related items into concise summaries
7. PRESERVE: The markdown structure (headers, bullet points)

The goal is a tight, focused file that gives this agent its operational context
without duplicating shared knowledge.

Current MEMORY.md:
---
{content}
---

Output the compacted MEMORY.md content only, no explanation.
```

### What Stays in MEMORY.md vs Graph

| Content | Where | Example |
|---------|-------|---------|
| Agent working style | MEMORY.md | "When making large edits, use file overwrite instead of in-place edits" |
| User preferences | Graph (shared) | "Alex prefers concise responses" |
| Project decisions | Graph (shared) | "We chose OAuth2 with PKCE for auth" |
| Active task context | MEMORY.md | "Currently working on evidence index implementation" |
| Tool/workflow preferences | MEMORY.md | "The command 'cod' is an alias for 'codex --yolo'" |
| Reference facts | Graph (shared) | "The nanobot fork repo is at github.com/akatz-ai/nanobot" |
| Architectural decisions | Graph (shared) | "Memory prepend architecture for prompt caching" |

### Safety

- Before overwriting, save the previous version: `MEMORY.md.bak` (single backup, overwritten each time)
- Log the before/after token counts
- If the compaction model returns empty or obviously broken output, keep the original

---

## Implementation Plan

### File Changes

**New files:**
| File | Purpose |
|------|---------|
| `nanobot/session/evidence.py` | `EvidenceIndex` class — SQLite FTS5 wrapper |
| `nanobot/agent/tools/session_search.py` | `SessionSearchTool` — agent-facing search tool |
| `nanobot/session/extraction.py` | `SessionExtractor` — periodic extraction logic |
| `nanobot/session/memory_compactor.py` | `MemoryMdCompactor` — MEMORY.md distillation |

**Modified files:**
| File | Changes |
|------|---------|
| `nanobot/agent/loop.py` | Add evidence indexing hook post-tool-execution, add periodic extraction trigger check, wire up MEMORY.md compaction |
| `nanobot/agent/tools/__init__.py` | Register `SessionSearchTool` |
| `nanobot/session/manager.py` | Add `ExtractionState` to session metadata helpers |
| `nanobot/session/compaction.py` | Add catch-up extraction call before cutting messages |

### Implementation Order

**Phase 1: Evidence Index** (can ship independently)
1. `evidence.py` — SQLite FTS5 wrapper with add/search/close
2. `session_search.py` — tool definition and result formatting
3. Hook in `loop.py` post-tool-execution
4. Seeding script for existing sessions
5. Tests: indexing, search, chunking, tool filtering

**Phase 2: Extraction Decoupling**
1. `extraction.py` — extraction watermark, trigger logic, batch planning
2. Modify `loop.py` — check extraction triggers after each user turn
3. Modify `compaction.py` — catch-up extraction before cut point
4. Update extraction prompt for shared/agent routing
5. Tests: watermark advancement, failure handling, catch-up, trigger conditions

**Phase 3: MEMORY.md Compaction**
1. `memory_compactor.py` — token counting, compaction prompt, safety checks
2. Wire into extraction runs in `loop.py`
3. Tests: compaction trigger, content preservation, backup creation

### Testing Strategy

**Unit tests:**
- Evidence: index 100 tool results, search finds correct matches, chunking works, tool filter works
- Extraction: watermark only advances on success, catch-up triggers correctly, failure doesn't lose data
- MEMORY.md: compaction triggers at threshold, preserves agent-specific content, removes shared facts

**Integration tests:**
- 50-turn conversation: evidence index grows, extraction runs at turn 20 and 40, MEMORY.md stays under threshold
- Compaction + extraction: messages aren't cut before extraction completes
- Session reload: evidence index persists and is searchable after restart
- `/new` command: triggers final extraction before clearing session

**Failure tests:**
- Extraction model returns garbage → watermark unchanged, retry next trigger
- SQLite corruption → evidence index recreated from JSONL on next access
- MEMORY.md compaction returns empty → original preserved

### Configuration

```yaml
memory:
  extraction:
    enabled: true
    trigger_every_user_turns: 20
    idle_flush_minutes: 30
    force_before_session_clear: true
    overlap_messages: 5          # backward context overlap between extraction windows

  evidence:
    enabled: true
    min_content_chars: 512       # minimum tool output size to index
    chunk_size: 1200             # FTS chunk size in characters
    chunk_overlap: 100           # overlap between chunks
    seed_on_startup: true        # backfill existing sessions

  memory_md:
    compact_threshold_tokens: 10000   # trigger compaction above this
    compact_target_tokens: 5000       # target size after compaction
    backup_before_compact: true       # save .bak before overwriting
```

---

## Open Questions

1. **Evidence retention:** Should we ever prune the evidence SQLite? For very long sessions (3000+ messages), the index could grow large. Probably fine — SQLite handles this well and disk is cheap.

2. **Cross-session evidence:** Should the `session_search` tool only search the current session, or optionally search other recent sessions for the same agent? Start with current-session-only, extend later.

3. **Evidence + memory graph interaction:** When extraction finds a fact in tool output, should it also reference the evidence row? This would let retrieval say "this fact was extracted from exec output at message #1847" — nice for provenance but adds complexity. Defer to Phase 2.

4. **MEMORY.md format:** Should we add frontmatter to MEMORY.md with metadata (last compacted, token count, version)? Useful for tooling but adds complexity to the always-in-context file.

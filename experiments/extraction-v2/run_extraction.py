#!/usr/bin/env python3
"""
Extraction v2 Experiment — test the full extraction pipeline in isolation.

Reads real session JSONL, calls Haiku to extract memories, deduplicates
against the existing memory graph, and reports what it would do.

Usage:
    python experiments/extraction-v2/run_extraction.py [--window-start N] [--window-size N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanobot.providers.anthropic_auth import get_oauth_token

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SESSION_JSONL = Path.home() / ".nanobot/workspace/agents/nanobot-dev/sessions/discord_1476048732343763189.jsonl"
HAIKU_MODEL = "claude-haiku-4-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Extraction window
DEFAULT_WINDOW_SIZE = 25        # messages to extract from
CONTEXT_OVERLAP = 5             # read-only context before extraction window
MAX_EXTRACTION_TOKENS = 4096    # max output tokens for extraction call

# Dedup thresholds
SIMILARITY_IDENTICAL = 0.95     # reinforce only
SIMILARITY_RELATED = 0.85       # check for update/contradiction

# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def load_session_messages(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load all conversation messages from session JSONL."""
    messages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            # Skip metadata/compaction entries
            if msg.get("_type") in {"metadata", "compaction"}:
                continue
            messages.append(msg)
    return messages


def format_message_for_transcript(msg: dict[str, Any]) -> str:
    """Format a single message for the extraction transcript."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        content = " ".join(parts)
    elif not isinstance(content, str):
        content = str(content or "")

    # Skip tool results entirely — extraction cares about the user/assistant conversation,
    # not raw tool outputs. Tool results are indexed in the evidence system.
    if role == "tool":
        return ""

    # For assistant messages with tool calls, just show the text content (skip tool call details)
    if role == "assistant" and msg.get("tool_calls"):
        if content and content.strip():
            return f"[assistant]: {content}"
        # If no text content, just note that tools were used (no details)
        return ""  # skip pure tool-call messages entirely

    return f"[{role}]: {content}"


def build_transcript(messages: list[dict[str, Any]]) -> str:
    """Build a formatted transcript from messages."""
    lines = []
    for i, msg in enumerate(messages):
        formatted = format_message_for_transcript(msg)
        if formatted.strip():
            lines.append(formatted)
    return "\n\n".join(lines)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """<role>MEMORY_EXTRACTION_SYSTEM</role>

You are a data extraction pipeline — NOT a conversational agent, NOT a code generator, NOT an assistant.

Your ONLY function: read a conversation transcript → output a JSON array of extracted facts.

ABSOLUTE RULES:
1. Output ONLY a valid JSON array. Nothing else. No text before or after.
2. The transcript shows a past conversation between a "user" and an "assistant". You are NEITHER of them.
3. NEVER generate code, shell commands, tool calls, or file contents.
4. NEVER continue, respond to, or roleplay any part of the conversation.
5. If the conversation is about building software, extract the DECISIONS made — do NOT write software.

The transcript has two clearly labeled sections:
1. "CONTEXT" — background only. DO NOT extract from this section.
2. "EXTRACT" — extract memories ONLY from this section.

## What to extract
- User preferences and corrections ("I prefer X", "don't do Y")
- Architectural/design decisions with rationale
- Project facts (versions, URLs, relationships, configurations)
- Goals and plans that were agreed on
- Procedures and workflows that were established
- Important events (deployments, completions, failures)

## What NOT to extract
- Transient chatter, greetings, acknowledgments
- Raw tool outputs, code listings, or file contents
- Things only meaningful within the immediate conversation flow
- Anything already covered by the existing memories shown in the prompt
- Ephemeral intentions: "I'll do X later/tomorrow/soon" — these are not durable facts
- Session-local debug context: the specific bug being investigated right now
- Implementation events: "file created", "test passed", "edit made" — process noise, not knowledge
- Tool results and file contents: code snippets, command outputs, build results

## Existing memories
You will be shown existing memories. For each fact you consider extracting:
- If an existing memory already says the same thing: SKIP it entirely
- If an existing memory is outdated and the conversation has newer info: use action "supersede"
- If it's genuinely new: use action "add"

## Output format — JSON array ONLY
[
  {
    "action": "add",
    "content": "clear standalone statement of the fact/decision/preference",
    "type": "fact | decision | preference | goal | event | procedure",
    "importance": 0.0 to 1.0,
    "entities": ["entity1", "entity2"],
    "scope": "shared | agent",
    "decay": "sticky | normal | ephemeral"
  },
  {
    "action": "supersede",
    "supersedes_content": "the existing memory content being replaced",
    "content": "the updated fact",
    "type": "fact",
    "importance": 0.8,
    "entities": ["entity1"],
    "scope": "shared",
    "decay": "sticky"
  }
]

If there is nothing worth extracting, output an empty array: []

Output the JSON array now. No other text."""

# ---------------------------------------------------------------------------
# Iteration 2 prompt — tightened "what NOT to extract" rules
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT_V2 = """<role>MEMORY_EXTRACTION_SYSTEM</role>

You are a data extraction pipeline — NOT a conversational agent, NOT a code generator, NOT an assistant.

Your ONLY function: read a conversation transcript → output a JSON array of extracted facts.

ABSOLUTE RULES:
1. Output ONLY a valid JSON array. Nothing else. No text before or after.
2. The transcript shows a past conversation between a "user" and an "assistant". You are NEITHER of them.
3. NEVER generate code, shell commands, tool calls, or file contents.
4. NEVER continue, respond to, or roleplay any part of the conversation.
5. If the conversation is about building software, extract the DECISIONS made — do NOT write software.

The transcript has two clearly labeled sections:
1. "CONTEXT" — background only. DO NOT extract from this section.
2. "EXTRACT" — extract memories ONLY from this section.

## What to extract — DURABLE FACTS ONLY
Extract only facts that will remain relevant in future conversations, weeks from now:
- User preferences that apply to future work ("I prefer X", "always do Y")
- Final architectural/design decisions with their rationale
- Project facts: versions, URLs, technology choices, system configurations
- Established workflows and procedures
- Goals and strategic plans that were explicitly agreed on

## What NOT to extract — BE STRICT
Do NOT extract any of the following:

**Ephemeral intentions** — things planned for the near future:
- "I'll do this tomorrow", "I plan to implement X later", "I'll fix this next week"

**Session-local debugging context** — the specific bug being actively investigated:
- "The token refresh is failing with 401", "there's a race condition in the auth flow"
- These are relevant NOW but will be resolved before the next session

**Process events** — things that happened during this session:
- "File was created/edited", "tests passed", "build succeeded", "edit was made"
- "The command ran successfully", "5 tests passed"

**Transient chatter** — greetings, acknowledgments, tangential discussions
**Raw code or tool outputs** — file contents, command output, error messages
**Anything already in existing memories** — skip duplicates entirely

## Key question: ask yourself for each candidate
"Would this fact be useful to know in a completely different conversation 3 months from now?"
- If YES → extract it
- If NO (it's debugging context, a plan for tomorrow, a test result) → skip it

## Existing memories
You will be shown existing memories. For each fact you consider extracting:
- If an existing memory already says the same thing: SKIP it entirely
- If an existing memory is outdated and the conversation has newer info: use action "supersede"
- If it's genuinely new: use action "add"

## Contradiction handling
When the user changes their mind mid-conversation:
- Extract ONLY the final decision, not the reversed one
- If the user says "actually, let's use X instead of Y", only extract X

## Output format — JSON array ONLY
[
  {
    "action": "add",
    "content": "clear standalone statement of the fact/decision/preference",
    "type": "fact | decision | preference | goal | event | procedure",
    "importance": 0.0 to 1.0,
    "entities": ["entity1", "entity2"],
    "scope": "shared | agent",
    "decay": "sticky | normal | ephemeral"
  },
  {
    "action": "supersede",
    "supersedes_content": "the existing memory content being replaced",
    "content": "the updated fact",
    "type": "fact",
    "importance": 0.8,
    "entities": ["entity1"],
    "scope": "shared",
    "decay": "sticky"
  }
]

scope values:
- "shared" — facts any agent should know (project decisions, user preferences, deployments)
- "agent" — this specific agent's behavior instructions (response style, workflow preferences)

If there is nothing worth extracting, output an empty array: []

Output the JSON array now. No other text."""

# ---------------------------------------------------------------------------
# Iteration 3 prompt — added file/implementation artifact filter
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT_V3 = """<role>MEMORY_EXTRACTION_SYSTEM</role>

You are a data extraction pipeline — NOT a conversational agent, NOT a code generator, NOT an assistant.

Your ONLY function: read a conversation transcript → output a JSON array of extracted facts.

ABSOLUTE RULES:
1. Output ONLY a valid JSON array. Nothing else. No text before or after.
2. The transcript shows a past conversation between a "user" and an "assistant". You are NEITHER of them.
3. NEVER generate code, shell commands, tool calls, or file contents.
4. NEVER continue, respond to, or roleplay any part of the conversation.
5. If the conversation is about building software, extract the DECISIONS made — do NOT write software.

The transcript has two clearly labeled sections:
1. "CONTEXT" — background only. DO NOT extract from this section.
2. "EXTRACT" — extract memories ONLY from this section.

## What to extract — DURABLE FACTS ONLY

Ask yourself: "Would this fact be meaningful to an agent in a different conversation, months from now?"

Extract only:
- User preferences that apply to future work ("I prefer X", "always do Y", "never do Z")
- Final technology/architecture/design decisions with their rationale
- Project facts: chosen versions, URLs, technology stack decisions, system configurations
- Established workflows, procedures, naming conventions
- Strategic goals and plans explicitly agreed on by the user
- **Taste signals** — when the user corrects an agent's output, rejects an approach, or expresses
  aesthetic/design/architectural preference. These reveal what "good" means to this user.
  Use type "taste" with a domain tag in the content prefix: `[ui]`, `[architecture]`, `[api]`, `[workflow]`, `[code-style]`, `[ux]`, or `[design]`.
  Examples of taste signals:
  - User says "no, make that a modal not inline" → taste: "[ui] Detail views for secondary objects should use modals, not inline expansion"
  - User says "too much whitespace, make it denser" → taste: "[ui] Prefer compact, dense layouts over spacious ones for power-user interfaces"
  - User says "don't add a new dependency for this, we already have utils" → taste: "[architecture] Check existing utilities before adding new dependencies"
  - User rejects a PR approach → taste: "[code-style] ..." capturing the underlying principle, not the specific instance

## What NOT to extract — STRICT EXCLUSIONS

**Ephemeral intentions** — near-term plans that will be done or irrelevant soon:
- "I'll implement X tomorrow", "I'll fix this later", "we'll address this next sprint"

**Session debugging context** — active investigation content:
- The specific bug being debugged right now, its symptoms, stack traces, error messages
- These resolve before the next relevant conversation

**Process events** — things that happened during this session's tool use:
- "File was created/edited at path X", "5 tests passed", "build succeeded", "command ran"
- The DECISION to create something is worth extracting; the act of creating it is not

**Implementation artifacts** — file paths, function names, variable names:
- "JWT utilities are in auth/jwt_utils.py" → NOT worth extracting (file could move/rename)
- "JWT utilities use RS256 with stateless verification" → worth extracting (architecture decision)
- "The auth config is in config.py" → NOT worth extracting
- "Auth requires MFA and 24-hour sessions" → worth extracting (configuration decision)

**Transient chatter** — greetings, acknowledgments, off-topic digressions
**Raw content** — code snippets, tool outputs, file contents, error messages
**Duplicates** — anything already in existing memories

## Contradiction handling
When the user changes their mind mid-conversation:
- Extract ONLY the final decision, not intermediate rejected options

## Existing memories
For each fact you consider extracting:
- Same content as existing memory: SKIP entirely
- Existing memory is outdated, conversation has newer info: use action "supersede"
- Genuinely new: use action "add"

## Output format — JSON array ONLY
[
  {
    "action": "add",
    "content": "clear standalone statement of the fact/decision/preference",
    "type": "fact | decision | preference | goal | event | procedure | taste",
    "importance": 0.0 to 1.0,
    "entities": ["entity1", "entity2"],
    "scope": "shared | agent",
    "decay": "sticky | normal | ephemeral"
  },
  {
    "action": "supersede",
    "supersedes_content": "the existing memory content being replaced",
    "content": "the updated fact",
    "type": "fact",
    "importance": 0.8,
    "entities": ["entity1"],
    "scope": "shared",
    "decay": "sticky"
  }
]

scope values:
- "shared" — project decisions, user preferences applicable to all agents, technical facts
- "agent" — behavior instructions specific to this agent (response style, workflow, review order)

If there is nothing worth extracting, output an empty array: []

Output the JSON array now. No other text."""


def build_extraction_prompt(
    context_messages: list[dict[str, Any]],
    extraction_messages: list[dict[str, Any]],
    existing_memories: list[dict[str, Any]],
) -> str:
    """Build the user prompt for extraction."""
    parts = []

    parts.append("<transcript>")

    # Context section
    if context_messages:
        context_transcript = build_transcript(context_messages)
        parts.append(f"<context_section>\n{context_transcript}\n</context_section>")

    # Extraction section
    extraction_transcript = build_transcript(extraction_messages)
    parts.append(f"<extraction_section>\n{extraction_transcript}\n</extraction_section>")

    parts.append("</transcript>")

    # Existing memories section
    if existing_memories:
        mem_lines = []
        for i, mem in enumerate(existing_memories, 1):
            content = mem.get("content", "")
            mtype = mem.get("memory_type", "unknown")
            importance = mem.get("importance", 0.5)
            mem_lines.append(f"  {i}. [{mtype}, importance={importance}] {content}")
        parts.append(f"<existing_memories>\n" + "\n".join(mem_lines) + "\n</existing_memories>")
    else:
        parts.append("<existing_memories>None</existing_memories>")

    parts.append("\nExtract memories from <extraction_section> only. Output JSON array:")

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

async def call_haiku(
    system_prompt: str,
    user_prompt: str,
    model: str = HAIKU_MODEL,
    max_tokens: int = MAX_EXTRACTION_TOKENS,
) -> dict[str, Any]:
    """Call Anthropic API directly with OAuth token."""
    token = get_oauth_token()
    if not token:
        raise RuntimeError("No OAuth token available. Run: claude login")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": "oauth-2025-04-20",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        t0 = time.monotonic()
        resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text}", file=sys.stderr)
            resp.raise_for_status()

        data = resp.json()
        usage = data.get("usage", {})

        return {
            "content": data["content"][0]["text"] if data.get("content") else "",
            "usage": usage,
            "elapsed_seconds": round(elapsed, 2),
            "model": data.get("model", model),
        }

# ---------------------------------------------------------------------------
# Memory graph query
# ---------------------------------------------------------------------------

def _build_memory_store():
    """Build a MemoryGraphStore using the same config as the running agent."""
    import os
    from agent_memory import LiteLLMEmbedding, MemoryGraphStore

    db_path = str(Path.home() / ".nanobot/memory.db")
    embedding = LiteLLMEmbedding(
        model="openai/text-embedding-3-small",
        dimensions=512,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return MemoryGraphStore(db_path=db_path, embedding=embedding)


async def query_existing_memories(
    extraction_transcript: str,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """Query the memory graph for existing memories related to the extraction window."""
    try:
        store = _build_memory_store()
        await store.initialize()

        # Extract key topics from the transcript for search
        # Use a simple heuristic: take first 500 chars as search query
        search_query = extraction_transcript[:500]

        results = await store.recall(
            query=search_query,
            mode="hybrid",
            agent_id="nanobot-dev",
            max_results=max_results,
            graph_depth=0,  # no graph expansion for speed
        )

        return results

    except Exception as e:
        print(f"  [!] Memory graph query failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return []

# ---------------------------------------------------------------------------
# Dedup analysis (post-extraction)
# ---------------------------------------------------------------------------

async def analyze_dedup(
    extracted_items: list[dict[str, Any]],
    existing_memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Analyze each extracted item against existing memories using embeddings."""
    try:
        import os
        import numpy as np
        from agent_memory import LiteLLMEmbedding

        embedder = LiteLLMEmbedding(
            model="openai/text-embedding-3-small",
            dimensions=512,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Embed existing memories
        existing_texts = [m.get("content", "") for m in existing_memories]
        existing_embeddings = []
        if existing_texts:
            existing_embeddings = await embedder.embed(existing_texts)

        results = []
        for item in extracted_items:
            item_embedding = (await embedder.embed([item["content"]]))[0]

            best_match = None
            best_similarity = 0.0

            for j, exist_emb in enumerate(existing_embeddings):
                # Cosine similarity
                sim = float(np.dot(item_embedding, exist_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(exist_emb) + 1e-8
                ))
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = existing_memories[j]

            dedup_action = "add"
            if best_similarity > SIMILARITY_IDENTICAL:
                dedup_action = "reinforce"
            elif best_similarity > SIMILARITY_RELATED:
                dedup_action = "review"  # needs human/LLM review

            results.append({
                **item,
                "dedup_action": dedup_action,
                "best_match_similarity": round(best_similarity, 4),
                "best_match_content": best_match.get("content", "") if best_match else None,
            })

        return results

    except Exception as e:
        print(f"  [!] Dedup analysis failed: {e}", file=sys.stderr)
        # Return items without dedup info
        return [{**item, "dedup_action": "add", "best_match_similarity": 0.0, "best_match_content": None} for item in extracted_items]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Extraction v2 experiment")
    parser.add_argument("--session", type=str, default=str(SESSION_JSONL),
                        help="Path to session JSONL")
    parser.add_argument("--window-start", type=int, default=None,
                        help="Start of extraction window (default: last N messages)")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                        help="Number of messages to extract from")
    parser.add_argument("--context-overlap", type=int, default=CONTEXT_OVERLAP,
                        help="Context messages before extraction window")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be sent without calling API")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip embedding-based dedup analysis")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    session_path = Path(args.session)
    print(f"Loading session: {session_path}")
    messages = load_session_messages(session_path)
    print(f"  Total messages: {len(messages)}")

    # Count user turns
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    print(f"  User turns: {user_turns}")

    # Determine extraction window
    if args.window_start is not None:
        extract_start = args.window_start
    else:
        extract_start = max(0, len(messages) - args.window_size)

    extract_end = min(len(messages), extract_start + args.window_size)
    context_start = max(0, extract_start - args.context_overlap)

    print(f"\n  Context window: messages[{context_start}:{extract_start}] ({extract_start - context_start} msgs)")
    print(f"  Extraction window: messages[{extract_start}:{extract_end}] ({extract_end - extract_start} msgs)")

    context_msgs = messages[context_start:extract_start]
    extraction_msgs = messages[extract_start:extract_end]

    # Show message roles in each window
    context_roles = [m.get("role", "?") for m in context_msgs]
    extract_roles = [m.get("role", "?") for m in extraction_msgs]
    print(f"  Context roles: {context_roles}")
    print(f"  Extraction roles: {extract_roles}")

    # Build transcript for the extraction window
    extraction_transcript = build_transcript(extraction_msgs)
    context_transcript = build_transcript(context_msgs)
    print(f"\n  Context transcript: {len(context_transcript)} chars")
    print(f"  Extraction transcript: {len(extraction_transcript)} chars")

    # Query existing memories
    print(f"\nQuerying memory graph for existing related memories...")
    existing_memories = await query_existing_memories(extraction_transcript)
    print(f"  Found {len(existing_memories)} existing memories")
    if existing_memories:
        print(f"  Top 5:")
        for i, mem in enumerate(existing_memories[:5], 1):
            content = mem.get("content", "")[:100]
            mtype = mem.get("memory_type", "?")
            print(f"    {i}. [{mtype}] {content}...")

    # Build the full prompt
    user_prompt = build_extraction_prompt(context_msgs, extraction_msgs, existing_memories)
    print(f"\n  Full prompt size: {len(user_prompt)} chars (~{len(user_prompt) // 4} tokens)")
    print(f"  System prompt size: {len(EXTRACTION_SYSTEM_PROMPT)} chars")

    if args.dry_run:
        print("\n--- DRY RUN: Would send the following ---")
        print(f"\n[SYSTEM PROMPT]\n{EXTRACTION_SYSTEM_PROMPT[:500]}...")
        print(f"\n[USER PROMPT — first 2000 chars]\n{user_prompt[:2000]}...")
        print(f"\n[USER PROMPT — last 1000 chars]\n...{user_prompt[-1000:]}")
        return

    # Call Haiku
    print(f"\nCalling {HAIKU_MODEL}...")
    result = await call_haiku(EXTRACTION_SYSTEM_PROMPT, user_prompt)
    print(f"  Model: {result['model']}")
    print(f"  Elapsed: {result['elapsed_seconds']}s")
    print(f"  Usage: {result['usage']}")

    # Parse extracted items
    raw_content = result["content"].strip()
    # Strip markdown fences if present
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        raw_content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        extracted_items = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"\n  [!] Failed to parse JSON: {e}")
        print(f"  Raw output:\n{raw_content[:2000]}")
        return

    if not isinstance(extracted_items, list):
        print(f"  [!] Expected JSON array, got: {type(extracted_items)}")
        return

    print(f"\n{'='*70}")
    print(f"EXTRACTED {len(extracted_items)} ITEMS")
    print(f"{'='*70}")

    for i, item in enumerate(extracted_items, 1):
        action = item.get("action", "add")
        content = item.get("content", "")
        mtype = item.get("type", "?")
        importance = item.get("importance", 0.5)
        scope = item.get("scope", "?")
        decay = item.get("decay", "?")
        entities = item.get("entities", [])

        action_symbol = {"add": "➕", "supersede": "🔄"}.get(action, "❓")
        print(f"\n  {action_symbol} [{i}] {action.upper()}")
        print(f"     Content: {content}")
        print(f"     Type: {mtype} | Importance: {importance} | Scope: {scope} | Decay: {decay}")
        if entities:
            print(f"     Entities: {', '.join(entities)}")
        if action == "supersede":
            print(f"     Supersedes: {item.get('supersedes_content', '?')}")

    # Dedup analysis
    if not args.skip_dedup and existing_memories:
        print(f"\n{'='*70}")
        print(f"DEDUP ANALYSIS (embedding similarity)")
        print(f"{'='*70}")

        deduped = await analyze_dedup(extracted_items, existing_memories)
        for i, item in enumerate(deduped, 1):
            dedup_action = item.get("dedup_action", "?")
            sim = item.get("best_match_similarity", 0.0)
            match_content = item.get("best_match_content", "")

            symbol = {"add": "✅", "reinforce": "🔁", "review": "⚠️"}.get(dedup_action, "❓")
            print(f"\n  {symbol} [{i}] {dedup_action.upper()} (similarity: {sim})")
            print(f"     Extracted: {item.get('content', '')[:100]}")
            if match_content:
                print(f"     Closest:   {match_content[:100]}")
    else:
        deduped = extracted_items

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total extracted: {len(extracted_items)}")
    by_action = {}
    for item in extracted_items:
        a = item.get("action", "add")
        by_action[a] = by_action.get(a, 0) + 1
    for action, count in sorted(by_action.items()):
        print(f"    {action}: {count}")

    by_type = {}
    for item in extracted_items:
        t = item.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"  By type: {by_type}")

    by_scope = {}
    for item in extracted_items:
        s = item.get("scope", "unknown")
        by_scope[s] = by_scope.get(s, 0) + 1
    print(f"  By scope: {by_scope}")

    if not args.skip_dedup and existing_memories:
        by_dedup = {}
        for item in deduped:
            d = item.get("dedup_action", "unknown")
            by_dedup[d] = by_dedup.get(d, 0) + 1
        print(f"  Dedup verdict: {by_dedup}")

    # Save results
    if args.output:
        output_data = {
            "session": str(session_path),
            "window": {"context_start": context_start, "extract_start": extract_start, "extract_end": extract_end},
            "existing_memories_count": len(existing_memories),
            "api_usage": result["usage"],
            "api_elapsed": result["elapsed_seconds"],
            "extracted_items": extracted_items,
            "deduped_items": deduped if not args.skip_dedup else None,
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

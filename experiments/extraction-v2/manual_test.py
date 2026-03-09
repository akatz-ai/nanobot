#!/usr/bin/env python3
"""
Manual test runner — runs specific scenarios through Haiku and prints detailed results.
Uses V3 prompt (the recommended production prompt).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_extraction import (
    EXTRACTION_SYSTEM_PROMPT_V3,
    build_extraction_prompt,
    build_transcript,
    call_haiku,
    format_message_for_transcript,
    load_session_messages,
)

SCENARIOS_DIR = Path(__file__).parent / "test_scenarios"
RESULTS_DIR = Path(__file__).parent / "results" / "manual"


def load_scenario(n: int):
    """Load a scenario's JSONL and memories."""
    jsonl_path = SCENARIOS_DIR / f"scenario_{n}.jsonl"
    mem_path = SCENARIOS_DIR / f"scenario_{n}_memories.json"
    desc_path = SCENARIOS_DIR / f"scenario_{n}_description.md"

    messages = load_session_messages(jsonl_path)
    memories = json.loads(mem_path.read_text()) if mem_path.exists() else []
    description = desc_path.read_text() if desc_path.exists() else ""

    return messages, memories, description


async def run_scenario(n: int, prompt_version: str = "v3"):
    """Run a single scenario and return results."""
    messages, memories, description = load_scenario(n)

    # Use all messages as extraction window, first 3 as context
    context_end = min(3, len(messages))
    context_msgs = messages[:context_end]
    extract_msgs = messages[context_end:]

    if not extract_msgs:
        extract_msgs = messages
        context_msgs = []

    # Build prompt
    user_prompt = build_extraction_prompt(context_msgs, extract_msgs, memories)

    # Show transcript that Haiku will see
    transcript = build_transcript(extract_msgs)

    print(f"\n{'='*70}")
    print(f"SCENARIO {n}")
    print(f"{'='*70}")
    print(f"Messages: {len(messages)} total, {len(context_msgs)} context, {len(extract_msgs)} extract")
    print(f"Existing memories: {len(memories)}")
    print(f"Transcript length: {len(transcript)} chars")

    if memories:
        print(f"\nExisting memories:")
        for i, m in enumerate(memories, 1):
            print(f"  {i}. [{m.get('memory_type','?')}] {m.get('content','')}")

    # Call Haiku
    system_prompt = EXTRACTION_SYSTEM_PROMPT_V3
    print(f"\nCalling Haiku (V3 prompt)...")
    result = await call_haiku(system_prompt, user_prompt)
    print(f"  Elapsed: {result['elapsed_seconds']}s")
    print(f"  Tokens: {result['usage'].get('input_tokens', 0)} in, {result['usage'].get('output_tokens', 0)} out")

    # Parse
    raw = result["content"].strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n  ❌ JSON PARSE ERROR: {e}")
        print(f"  Raw output (first 500 chars):\n{raw[:500]}")
        return None

    if not isinstance(items, list):
        print(f"\n  ❌ Expected array, got {type(items)}")
        return None

    print(f"\n  Extracted {len(items)} items:")
    for i, item in enumerate(items, 1):
        action = item.get("action", "add")
        content = item.get("content", "")
        mtype = item.get("type", "?")
        importance = item.get("importance", 0.5)
        scope = item.get("scope", "?")
        decay = item.get("decay", "?")
        entities = item.get("entities", [])

        symbol = {"add": "➕", "supersede": "🔄"}.get(action, "❓")
        print(f"\n  {symbol} [{i}] {action.upper()}")
        print(f"     {content}")
        print(f"     type={mtype} importance={importance} scope={scope} decay={decay}")
        if entities:
            print(f"     entities: {entities}")
        if action == "supersede":
            print(f"     supersedes: {item.get('supersedes_content', '?')}")

    return {
        "scenario": n,
        "items": items,
        "usage": result["usage"],
        "elapsed": result["elapsed_seconds"],
        "raw": raw,
    }


async def main():
    scenarios = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [3, 4, 5, 6, 8, 9, 10]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for n in scenarios:
        try:
            result = await run_scenario(n)
            if result:
                all_results[n] = result
        except Exception as e:
            print(f"\n  ❌ SCENARIO {n} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save all results
    output = RESULTS_DIR / "manual_results.json"
    output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n\nResults saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())

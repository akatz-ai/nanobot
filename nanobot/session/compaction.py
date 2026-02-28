"""Structured session compaction helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.session.manager import CompactionEntry, Session


SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "and produce a structured summary. Do NOT continue the conversation. "
    "ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """Produce a structured summary in this exact format:

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

Preserve exact file paths, function names, error messages, and version numbers.
Keep the summary concise: under 1200 words and under 6000 characters.
Collapse stale completed work into grouped bullets when not needed for next steps."""

UPDATE_SUMMARIZATION_PROMPT = """Update the previous structured summary using the new conversation.

Requirements:
- Preserve all existing important information.
- Add new progress from this transcript.
- Move completed items from In Progress to Done.
- Refresh Next Steps to reflect current state.
- Keep the same section structure.
- Keep the summary concise: under 1200 words and under 6000 characters.
- Collapse stale completed work into grouped bullets when not needed for next steps.

Previous summary:
<previous_summary>
{previous_summary}
</previous_summary>
"""

SUMMARY_COMPRESSION_PROMPT = """Rewrite the previous structured summary into a tighter version.

Requirements:
- Keep the same section headers and structure.
- Preserve exact file paths, function names, error messages, and versions that still matter.
- Collapse stale completed work into short grouped bullets.
- Remove details that are no longer needed to safely continue.
- Keep output under {target_chars} characters.

Previous summary:
<previous_summary>
{previous_summary}
</previous_summary>
"""

TURN_PREFIX_SUMMARIZATION_PROMPT = """Summarize this partial turn prefix that was compacted mid-turn.

Output compact bullet points covering:
- User intent in this turn
- Tool actions already taken
- Intermediate findings
- What still remains in the turn

Preserve exact file paths, function names, and error messages.
"""

MAX_PREVIOUS_SUMMARY_CHARS = 8_000
TARGET_PREVIOUS_SUMMARY_CHARS = 4_000
MAX_SUMMARY_CHARS = 6_000
TARGET_SUMMARY_CHARS = 4_000
MAX_SUMMARY_COMPRESSION_INPUT_CHARS = 20_000
SUMMARY_COMPRESSION_MAX_TOKENS = 1_600


@dataclass
class CutPointResult:
    first_kept_index: int
    is_split_turn: bool
    turn_start_index: int | None
    tokens_kept: int
    tokens_cut: int


@dataclass
class CompactionPlan:
    summary_start: int
    summary_end: int
    first_kept_index: int
    extract_start: int
    extract_end: int
    is_split_turn: bool
    turn_start_index: int | None
    previous_summary: str | None


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        name = tool_call.get("name")
        return str(name) if name else "unknown_tool"
    name = getattr(tool_call, "name", None)
    return str(name) if name else "unknown_tool"


def _tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        tc_id = tool_call.get("id")
        if tc_id:
            return str(tc_id)
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("id"):
            return str(function["id"])
        return None
    tc_id = getattr(tool_call, "id", None)
    return str(tc_id) if tc_id else None


def _tool_call_args(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict) and "arguments" in function:
            raw = function.get("arguments")
        else:
            raw = tool_call.get("arguments", {})
    else:
        raw = getattr(tool_call, "arguments", {})

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
        return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_content_text(content: Any, *, role: str) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            if item_type in {"image_url", "input_image", "image"}:
                parts.append("[image attached]" if role == "user" else "[image]")
        return " ".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def serialize_conversation(
    messages: list[dict[str, Any]],
    max_transcript_chars: int = 100_000,
) -> str:
    """Serialize chat/tool history to a readable transcript."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown").lower()
        content_text = _extract_content_text(msg.get("content"), role=role)

        if role == "user":
            lines.append(f"[User]: {content_text}" if content_text else "[User]:")
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if content_text:
                lines.append(f"[Assistant]: {content_text}")
            if tool_calls:
                call_names = ", ".join(_tool_call_name(tc) for tc in tool_calls)
                lines.append(f"[Assistant]: (tool calls: {call_names})")
            elif not content_text:
                lines.append("[Assistant]:")
            continue

        if role == "tool":
            name = msg.get("name") or "tool"
            lines.append(f"[Tool:{name}]: {content_text}" if content_text else f"[Tool:{name}]:")
            continue

        pretty_role = role.capitalize()
        lines.append(f"[{pretty_role}]: {content_text}" if content_text else f"[{pretty_role}]:")

    transcript = "\n".join(lines)
    if len(transcript) <= max_transcript_chars:
        return transcript

    omitted = len(transcript) - max_transcript_chars
    tail = transcript[-max_transcript_chars:]
    marker = f"[... truncated {omitted} chars from beginning ...]\n"
    if len(marker) >= max_transcript_chars:
        return tail
    return marker + tail[len(marker):]


def _append_unique(target: list[str], value: str | None) -> None:
    if not value:
        return
    if value not in target:
        target.append(value)


def _extract_path_from_args(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filename", "file"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_exec_modified_paths(command: str) -> list[str]:
    # Best-effort capture of common write patterns from shell commands.
    patterns = [
        r"(?<!\d)>{1,2}\s*([\w./~\\-]+)",
        r"\btee\s+([\w./~\\-]+)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, command):
            _append_unique(matches, match)
    return matches


def extract_file_ops(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract file read/modify operations from successful tool calls."""
    read_files: list[str] = []
    modified_files: list[str] = []

    completed_tool_call_ids = {
        str(msg.get("tool_call_id"))
        for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            tc_id = _tool_call_id(tool_call)
            if not tc_id or tc_id not in completed_tool_call_ids:
                continue

            tool_name = _tool_call_name(tool_call)
            args = _tool_call_args(tool_call)
            path = _extract_path_from_args(args)

            if tool_name in {"read_file", "list_dir"}:
                _append_unique(read_files, path)
            elif tool_name in {"write_file", "edit_file"}:
                _append_unique(modified_files, path)
            elif tool_name == "exec":
                command = args.get("command") or args.get("cmd") or args.get("script")
                if isinstance(command, str):
                    for candidate in _extract_exec_modified_paths(command):
                        _append_unique(modified_files, candidate)

    return {"read_files": read_files, "modified_files": modified_files}


def _summary_has_required_sections(summary: str) -> bool:
    required = ["## Goal", "## Progress", "## Next Steps"]
    return all(section in summary for section in required)


def _fallback_summary(transcript: str, previous_summary: str | None = None) -> str:
    excerpt = transcript[-2000:] if transcript else "(empty transcript)"
    previous_block = (
        f"- Previous summary retained:\n{previous_summary[:1000]}"
        if previous_summary
        else "- None"
    )
    return (
        "## Goal\n"
        "Recover context after compaction fallback.\n\n"
        "## Constraints & Preferences\n"
        "- Preserve exact technical identifiers when possible.\n\n"
        "## Progress\n"
        "### Done\n"
        "- [x] Generated fallback summary because model output was malformed.\n"
        "### In Progress\n"
        "- [ ] Validate and refine this summary in the next compaction cycle.\n"
        "### Blocked\n"
        "- LLM summary output did not match the required structure.\n\n"
        "## Key Decisions\n"
        "- **Fallback summary used**: preserves continuity over strict formatting failure.\n\n"
        "## Next Steps\n"
        "1. Continue the task using this fallback context.\n"
        "2. Re-run compaction summary generation on the next cycle.\n\n"
        "## Critical Context\n"
        f"{previous_block}\n"
        f"- Transcript excerpt:\n{excerpt}"
    )


def _clip_from_start(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    head = f"[... truncated {omitted} chars from beginning ...]\n"
    if len(head) >= max_chars:
        return text[-max_chars:]
    return head + text[-(max_chars - len(head)) :]


def _hard_cap_summary(summary: str, max_chars: int) -> str:
    if len(summary) <= max_chars:
        return summary
    marker = "\n\n[... summary trimmed to stay within compaction budget ...]"
    keep = max(0, max_chars - len(marker))
    if keep == 0:
        return summary[:max_chars]
    return summary[:keep].rstrip() + marker


async def _compress_summary_to_target(
    summary: str,
    provider: LLMProvider,
    model: str,
    *,
    target_chars: int,
    reason: str,
) -> str:
    clipped_input = _clip_from_start(summary, MAX_SUMMARY_COMPRESSION_INPUT_CHARS)
    prompt = SUMMARY_COMPRESSION_PROMPT.format(
        previous_summary=clipped_input,
        target_chars=target_chars,
    )

    attempts = 2
    for attempt in range(attempts):
        response = await provider.chat(
            messages=[
                {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=SUMMARY_COMPRESSION_MAX_TOKENS,
        )
        content = (response.content or "").strip()
        if not _summary_has_required_sections(content):
            logger.warning(
                "Summary compression malformed (reason={}, attempt {}/{}): {}",
                reason,
                attempt + 1,
                attempts,
                content[:200],
            )
            continue
        if len(content) > target_chars:
            logger.warning(
                "Summary compression still over target (reason={}, attempt {}/{}): len={} target={}",
                reason,
                attempt + 1,
                attempts,
                len(content),
                target_chars,
            )
            continue
        return content

    logger.warning(
        "Summary compression fallback triggered (reason={}): source_len={} target={}",
        reason,
        len(summary),
        target_chars,
    )
    fallback = _fallback_summary("", previous_summary=clipped_input)
    return _hard_cap_summary(fallback, target_chars)


async def generate_compaction_summary(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
    previous_summary: str | None = None,
    max_transcript_chars: int = 100_000,
) -> str:
    """Generate a structured compaction summary from conversation history."""
    transcript = serialize_conversation(messages, max_transcript_chars=max_transcript_chars)
    effective_previous_summary = previous_summary
    if (
        isinstance(effective_previous_summary, str)
        and len(effective_previous_summary) > MAX_PREVIOUS_SUMMARY_CHARS
    ):
        logger.warning(
            "Previous compaction summary exceeds cap: len={} cap={}. Compressing before update.",
            len(effective_previous_summary),
            MAX_PREVIOUS_SUMMARY_CHARS,
        )
        effective_previous_summary = await _compress_summary_to_target(
            effective_previous_summary,
            provider,
            model,
            target_chars=TARGET_PREVIOUS_SUMMARY_CHARS,
            reason="previous summary over cap",
        )

    prompt_parts = [f"<conversation>\n{transcript}\n</conversation>"]
    if effective_previous_summary:
        prompt_parts.append(
            UPDATE_SUMMARIZATION_PROMPT.format(previous_summary=effective_previous_summary)
        )
    prompt_parts.append(SUMMARIZATION_PROMPT)
    prompt = "\n\n".join(prompt_parts)

    attempts = 2
    for attempt in range(attempts):
        response = await provider.chat(
            messages=[
                {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=3000,
        )
        content = (response.content or "").strip()
        if _summary_has_required_sections(content):
            if len(content) > MAX_SUMMARY_CHARS:
                logger.warning(
                    "Compaction summary exceeds cap: len={} cap={}. Compressing output.",
                    len(content),
                    MAX_SUMMARY_CHARS,
                )
                content = await _compress_summary_to_target(
                    content,
                    provider,
                    model,
                    target_chars=TARGET_SUMMARY_CHARS,
                    reason="generated summary over cap",
                )
            if len(content) > MAX_SUMMARY_CHARS:
                content = _hard_cap_summary(content, MAX_SUMMARY_CHARS)
            return content
        logger.warning(
            "Compaction summary malformed (attempt {}/{}): {}",
            attempt + 1,
            attempts,
            content[:200],
        )

    fallback = _fallback_summary(transcript, previous_summary=effective_previous_summary)
    return _hard_cap_summary(fallback, MAX_SUMMARY_CHARS)


async def generate_turn_prefix_summary(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str,
    max_transcript_chars: int = 20_000,
) -> str:
    """Generate a compact summary for a split-turn prefix segment."""
    transcript = serialize_conversation(messages, max_transcript_chars=max_transcript_chars)
    prompt = (
        f"<conversation>\n{transcript}\n</conversation>\n\n"
        f"{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
    response = await provider.chat(
        messages=[
            {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.0,
        max_tokens=1200,
    )
    content = (response.content or "").strip()
    if content:
        return content
    return "- Unable to summarize split-turn prefix; proceed using remaining turn context."


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Conservative token estimate for a single message."""
    total = 4  # minimum role/format overhead

    content = msg.get("content")
    if isinstance(content, str):
        total += max(0, len(content) // 3)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                total += max(0, len(item) // 3)
            elif isinstance(item, dict):
                item_type = item.get("type")
                text = item.get("text")
                if item_type in {"image_url", "input_image", "image"}:
                    total += 1200
                elif isinstance(text, str):
                    total += max(0, len(text) // 3)
                else:
                    total += max(0, len(json.dumps(item, ensure_ascii=False)) // 3)
    elif content is not None:
        total += max(0, len(str(content)) // 3)

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        total += max(0, len(json.dumps(tool_calls, ensure_ascii=False)) // 3)

    return max(4, total)


def find_valid_cut_points(
    messages: list[dict[str, Any]],
    after_index: int = 0,
) -> list[int]:
    """Return safe first-kept indices (turn boundaries) for compaction."""
    floor = max(0, int(after_index))
    pending_tool_calls: set[str] = set()
    points: list[int] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = _tool_call_id(tc)
                if tc_id:
                    pending_tool_calls.add(tc_id)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                pending_tool_calls.discard(str(tc_id))
        elif role == "user" and idx >= floor and not pending_tool_calls:
            points.append(idx)

    return points


def _find_turn_start(messages: list[dict[str, Any]], index: int, after_index: int) -> int | None:
    for pos in range(min(index, len(messages) - 1), -1, -1):
        if messages[pos].get("role") == "user":
            return pos
    if 0 <= after_index < len(messages):
        return after_index
    return None


def find_cut_point(
    messages: list[dict[str, Any]],
    keep_recent_tokens: int = 20_000,
    after_index: int = 0,
) -> CutPointResult | None:
    """Find a turn-aware compaction cut point for the given token budget."""
    if not messages:
        return None

    floor = max(0, int(after_index))
    if floor >= len(messages):
        return None

    estimates = [estimate_message_tokens(msg) for msg in messages]

    running = 0
    raw_index: int | None = None
    for idx in range(len(messages) - 1, floor - 1, -1):
        running += estimates[idx]
        if running >= keep_recent_tokens:
            raw_index = idx
            break

    # Under budget after previous compaction boundary.
    if raw_index is None:
        return None

    valid_points = [i for i in find_valid_cut_points(messages, after_index=floor) if i > floor]
    if raw_index <= floor and not valid_points:
        return None

    before_points = [i for i in valid_points if i <= raw_index]
    after_points = [i for i in valid_points if i >= raw_index]

    role = str(messages[raw_index].get("role") or "")
    is_tool_chain_node = role == "tool" or (
        role == "assistant" and bool(messages[raw_index].get("tool_calls"))
    )

    chosen_index: int | None = None
    is_split_turn = False
    turn_start_index: int | None = None

    # If the raw point is inside an active tool chain, favor moving forward to
    # the next clean boundary so we don't keep dangling tool messages.
    if is_tool_chain_node and after_points:
        chosen_index = after_points[0]
    elif before_points:
        chosen_index = before_points[-1]
    elif after_points:
        chosen_index = after_points[0]
    else:
        # Very long single turn: no clean boundary to cut on.
        is_split_turn = True
        chosen_index = max(floor + 1, raw_index)
        turn_start_index = _find_turn_start(messages, raw_index, floor)

    if chosen_index is None or chosen_index <= floor:
        return None

    tokens_kept = sum(estimates[chosen_index:])
    tokens_cut = sum(estimates[:chosen_index])

    return CutPointResult(
        first_kept_index=chosen_index,
        is_split_turn=is_split_turn,
        turn_start_index=turn_start_index,
        tokens_kept=tokens_kept,
        tokens_cut=tokens_cut,
    )


def should_compact(
    messages: list[dict[str, Any]],
    context_window: int = 200_000,
    reserve_tokens: int = 16_384,
    last_input_tokens: int | None = None,
) -> bool:
    """Decide whether compaction should run based on token pressure."""
    threshold = (context_window - reserve_tokens) * 0.7
    if last_input_tokens is not None:
        return float(last_input_tokens) > float(threshold)

    estimated_total = sum(estimate_message_tokens(msg) for msg in messages)
    return float(estimated_total) > float(threshold)


def _usage_snapshot_tokens(session: Session) -> int | None:
    raw = session.metadata.get("usage_snapshot")
    if not isinstance(raw, dict):
        return None
    value = raw.get("total_input_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def compact_session(
    session: Session,
    provider: LLMProvider,
    model: str,
    context_window: int = 200_000,
    reserve_tokens: int = 16_384,
    keep_recent_tokens: int = 20_000,
) -> CompactionEntry | None:
    """Run structured compaction and persist a CompactionEntry when successful."""
    last_input_tokens = _usage_snapshot_tokens(session)
    baseline_messages = session.get_history(
        max_messages=max(1, len(session.messages)),
        prune_tool_results=False,
        context_window=context_window,
    )
    baseline_tokens = sum(estimate_message_tokens(msg) for msg in baseline_messages)
    pressure_messages = session.get_history(
        max_messages=max(1, len(session.messages)),
        context_window=context_window,
    )
    post_prune_tokens = sum(estimate_message_tokens(msg) for msg in pressure_messages)
    pruning_applied = post_prune_tokens < baseline_tokens
    decision_tokens = (
        post_prune_tokens
        if pruning_applied
        else (
            int(last_input_tokens)
            if last_input_tokens is not None
            else post_prune_tokens
        )
    )
    if not should_compact(
        pressure_messages,
        context_window=context_window,
        reserve_tokens=reserve_tokens,
        last_input_tokens=decision_tokens,
    ):
        return None

    previous = session.get_last_compaction()
    after_index = previous.first_kept_index if previous else 0
    cut_point = find_cut_point(
        session.messages,
        keep_recent_tokens=keep_recent_tokens,
        after_index=after_index,
    )
    if cut_point is None:
        return None

    plan = CompactionPlan(
        summary_start=after_index,
        summary_end=cut_point.first_kept_index,
        first_kept_index=cut_point.first_kept_index,
        extract_start=after_index,
        extract_end=cut_point.first_kept_index,
        is_split_turn=cut_point.is_split_turn,
        turn_start_index=cut_point.turn_start_index,
        previous_summary=previous.summary if previous else None,
    )

    if plan.summary_end <= plan.summary_start:
        return None

    summary_messages = list(session.messages[plan.summary_start:plan.summary_end])
    try:
        summary = await generate_compaction_summary(
            summary_messages,
            provider,
            model,
            previous_summary=plan.previous_summary,
        )
        if plan.is_split_turn and plan.turn_start_index is not None:
            if plan.turn_start_index < plan.first_kept_index:
                split_prefix_messages = session.messages[
                    plan.turn_start_index: plan.first_kept_index
                ]
                split_prefix = await generate_turn_prefix_summary(
                    split_prefix_messages,
                    provider,
                    model,
                )
                summary = f"{summary}\n\n## Split Turn Prefix\n{split_prefix}"
    except Exception:
        logger.exception("Structured compaction summary generation failed for {}", session.key)
        # Avoid immediate retrigger loops from stale usage readings.
        session.metadata.pop("usage_snapshot", None)
        return None

    file_ops = extract_file_ops(summary_messages)
    tokens_before = int(decision_tokens)

    entry = session.append_compaction(
        summary=summary,
        first_kept_index=plan.first_kept_index,
        tokens_before=tokens_before,
        file_ops=file_ops,
        previous_summary=plan.previous_summary,
    )

    session.metadata["_structured_compaction_plan"] = {
        "summary_start": plan.summary_start,
        "summary_end": plan.summary_end,
        "extract_start": plan.extract_start,
        "extract_end": plan.extract_end,
        "cut_point_type": "split_turn" if plan.is_split_turn else "clean",
    }

    return entry

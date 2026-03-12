"""Per-call token usage logger.

Records API token usage for every LLM call as a sidecar JSONL file
alongside the session. Tracks input/output/cache tokens, cumulative
totals, and context window utilization.

Write-only from the agent loop; read-only from dashboards and CLI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class TokenUsageLogger:
    """Append-only sidecar log of per-call token usage.

    For each API call, we record:
      - turn: which user turn this belongs to (incremented per user message)
      - iteration: which iteration within the turn (1-based, increments per tool loop)
      - input_tokens: prompt tokens for this call
      - output_tokens: completion tokens for this call
      - cache_read_tokens: tokens served from prompt cache
      - cache_creation_tokens: tokens written to prompt cache
      - context_window: model's max context window
      - utilization_pct: input_tokens / context_window as percentage
      - cumulative_input: running total of input tokens across all calls in session
      - cumulative_output: running total of output tokens across all calls in session
      - model: model name used for this call
    """

    def __init__(self, session_path: Path) -> None:
        self._path = session_path.with_suffix(".usage.jsonl")
        self._turn = 0
        self._cumulative_input = 0
        self._cumulative_output = 0
        self._call_count = 0
        # Restore state from existing log
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            self._turn = entry.get("turn", 0)
                            self._cumulative_input = entry.get("cumulative_input", 0)
                            self._cumulative_output = entry.get("cumulative_output", 0)
                            self._call_count = entry.get("call_index", 0) + 1
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

    def new_turn(self) -> None:
        """Signal the start of a new user turn."""
        self._turn += 1

    def log_usage(
        self,
        *,
        usage: dict[str, int],
        iteration: int,
        context_window: int,
        model: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Record token usage from one API call.

        Args:
            usage: The usage dict from LLMResponse (prompt_tokens,
                completion_tokens, cache_read_input_tokens, etc.)
            iteration: Iteration number within the current turn (1-based).
            context_window: Model's max context window size in tokens.
            model: Model name used for this call.
            finish_reason: How the response ended (stop, length, tool_use, etc.)
        """
        try:
            self._write_entry(
                usage=usage,
                iteration=iteration,
                context_window=context_window,
                model=model,
                finish_reason=finish_reason,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to write usage log entry to {}", self._path
            )

    def _write_entry(
        self,
        *,
        usage: dict[str, int],
        iteration: int,
        context_window: int,
        model: str | None,
        finish_reason: str | None,
    ) -> None:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        # Anthropic's input_tokens only reports non-cached tokens.
        # Total context = input_tokens + cache_read + cache_creation.
        total_input = input_tokens + cache_read + cache_creation

        self._cumulative_input += total_input
        self._cumulative_output += output_tokens

        utilization = (total_input / context_window * 100) if context_window > 0 else 0

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_index": self._call_count,
            "turn": self._turn,
            "iteration": iteration,
            "model": model,
            "finish_reason": finish_reason,
            # Per-call token counts (total = all input including cached)
            "input_tokens": input_tokens,
            "total_input_tokens": total_input,
            "output_tokens": output_tokens,
            "total_tokens": total_input + output_tokens,
            # Cache breakdown
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "cache_hit_pct": round(cache_read / total_input * 100, 1) if total_input > 0 else 0,
            # Context window
            "context_window": context_window,
            "utilization_pct": round(utilization, 1),
            # Session cumulative
            "cumulative_input": self._cumulative_input,
            "cumulative_output": self._cumulative_output,
            "cumulative_total": self._cumulative_input + self._cumulative_output,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._call_count += 1

    @property
    def cumulative_input(self) -> int:
        return self._cumulative_input

    @property
    def cumulative_output(self) -> int:
        return self._cumulative_output

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        """Reset cumulative counters (e.g. after compaction)."""
        self._cumulative_input = 0
        self._cumulative_output = 0


def load_usage_log(session_path: Path) -> list[dict[str, Any]]:
    """Read all entries from a usage log file.

    Args:
        session_path: Path to the session JSONL (the ``.usage.jsonl``
            sibling will be read).

    Returns:
        List of usage log entries, ordered by call_index.
    """
    log_path = session_path.with_suffix(".usage.jsonl")
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda e: e.get("call_index", 0))
    return entries


def get_session_summary(session_path: Path) -> dict[str, Any] | None:
    """Get a summary of token usage for a session.

    Returns a dict with total tokens, average utilization, cache stats, etc.
    Returns None if no usage log exists.
    """
    entries = load_usage_log(session_path)
    if not entries:
        return None

    last = entries[-1]
    total_input = last.get("cumulative_input", 0)
    total_output = last.get("cumulative_output", 0)

    # Calculate average cache hit rate (clamp to 0-100 to handle old corrupted entries)
    cache_hits = [min(e.get("cache_hit_pct", 0), 100.0) for e in entries]
    avg_cache_hit = sum(cache_hits) / len(cache_hits) if cache_hits else 0

    # Peak utilization
    peak_util = max(e.get("utilization_pct", 0) for e in entries)

    return {
        "total_calls": len(entries),
        "total_turns": last.get("turn", 0),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "current_utilization_pct": last.get("utilization_pct", 0),
        "peak_utilization_pct": peak_util,
        "avg_cache_hit_pct": round(avg_cache_hit, 1),
        "context_window": last.get("context_window", 0),
        "model": last.get("model"),
    }

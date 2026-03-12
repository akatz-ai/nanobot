"""Canonical builders for session-scoped context inspection payloads."""

from __future__ import annotations

from typing import Any

from nanobot.session.compaction_log import load_compaction_log
from nanobot.session.context_log import load_context_log

_DEFAULT_CONTEXT_WINDOW = 200_000
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-3-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}
_CURRENT_CONTEXT_SNAPSHOT_SOURCES = {
    "provider_usage",
    "estimated_current_prompt",
    "recomputed_current_context",
}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def context_window_for_model(
    model: str | None,
    usage_summary: dict[str, Any] | None = None,
) -> int:
    usage_window = _coerce_int((usage_summary or {}).get("context_window"), 0)
    if usage_window > 0:
        return usage_window

    model_lower = (model or "").lower()
    for key, size in _MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return size
    return _DEFAULT_CONTEXT_WINDOW


def session_metadata_blob(meta: dict[str, Any]) -> dict[str, Any]:
    value = meta.get("metadata")
    if isinstance(value, dict):
        return value
    if isinstance(meta, dict):
        return meta
    return {}


def current_context_tokens(
    meta: dict[str, Any],
    usage_summary: dict[str, Any],
    context_window: int,
) -> int:
    _ = usage_summary
    _ = context_window
    usage_snapshot = session_metadata_blob(meta).get("usage_snapshot")
    if not isinstance(usage_snapshot, dict):
        return 0
    source = usage_snapshot.get("source")
    if isinstance(source, str) and source not in _CURRENT_CONTEXT_SNAPSHOT_SOURCES:
        return 0
    snapshot_tokens = _coerce_int(usage_snapshot.get("total_input_tokens"), 0)
    return snapshot_tokens if snapshot_tokens > 0 else 0


def _compaction_state_for_message_index(
    compaction_events: list[dict[str, Any]],
    user_message_index: int,
) -> dict[str, Any] | None:
    candidates = []
    for event in compaction_events:
        post_context = event.get("post_context")
        if not isinstance(post_context, dict):
            continue
        first_kept = _coerce_int(post_context.get("first_kept_index"), -1)
        if 0 <= first_kept <= user_message_index:
            candidates.append(event)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda event: _coerce_int(
            (event.get("post_context") or {}).get("first_kept_index"),
            -1,
        ),
    )


def _snapshot_from_payload(
    payload: dict[str, Any] | None,
    fallback_id: str | None = None,
    *,
    assembled_prompt_tokens: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    snapshot_id = (
        payload.get("snapshotId")
        or payload.get("snapshot_id")
        or payload.get("source")
        or fallback_id
    )
    return {
        "snapshotId": str(snapshot_id or fallback_id or "snapshot"),
        "assembledPromptTokens": _coerce_int(
            payload.get("assembled_prompt_tokens"),
            assembled_prompt_tokens,
        ),
        "stablePrefixTokens": _coerce_int(payload.get("stable_cached_prefix_tokens"), 0),
        "dynamicTurnTokens": _coerce_int(payload.get("dynamic_turn_tokens"), 0),
        "visibleConversationSliceTokens": _coerce_int(
            payload.get("visible_conversation_slice_tokens"),
            0,
        ),
    }


def _canonical_section_kind(kind: str) -> str:
    if kind == "memory_md":
        return "memory"
    if kind == "session_summary":
        return "compaction_summary"
    if kind in {"current_user", "history_user", "history_assistant", "history_tool"}:
        return "conversation"
    if kind in {"system_base", "resume_notice"}:
        return "system"
    if kind == "system_other":
        return "session_metadata"
    if "retrieved" in kind:
        return "retrieved_memory"
    if "current" in kind:
        return "session_metadata"
    return "conversation"


def build_prompt_assembly_payload(
    turn: dict[str, Any] | None,
    *,
    metadata: dict[str, Any],
    current_tokens: int,
    context_window: int,
    compaction_events: list[dict[str, Any]],
    user_message_index: int | None,
) -> dict[str, Any] | None:
    if not isinstance(turn, dict):
        return None

    prompt_assembly = turn.get("prompt_assembly")
    if not isinstance(prompt_assembly, dict):
        return None

    budget = prompt_assembly.get("budget")
    budget = budget if isinstance(budget, dict) else {}
    estimated_total_tokens = _coerce_int(
        prompt_assembly.get("estimated_total_tokens"),
        current_tokens,
    )
    pre_snapshot = _snapshot_from_payload(
        prompt_assembly.get("pre_compaction_snapshot")
        if isinstance(prompt_assembly.get("pre_compaction_snapshot"), dict)
        else None,
        fallback_id="pre_compaction",
        assembled_prompt_tokens=estimated_total_tokens,
    )
    if pre_snapshot is None:
        pre_snapshot = {
            "snapshotId": "pre_compaction",
            "assembledPromptTokens": estimated_total_tokens,
            "stablePrefixTokens": 0,
            "dynamicTurnTokens": 0,
            "visibleConversationSliceTokens": 0,
        }

    metadata_blob = session_metadata_blob(metadata)
    trigger_snapshot = str(
        (metadata_blob.get("compaction_trigger_snapshot") or {}).get("trigger_snapshot")
        or "pre_compaction"
    )
    post_snapshot = _snapshot_from_payload(
        metadata_blob.get("post_compaction_snapshot")
        if isinstance(metadata_blob.get("post_compaction_snapshot"), dict)
        else None,
        fallback_id="post_compaction",
    )
    if user_message_index is not None:
        compaction_state = _compaction_state_for_message_index(
            compaction_events,
            user_message_index,
        )
        if isinstance(compaction_state, dict):
            pre_context = compaction_state.get("pre_context") or {}
            pre_prompt_snapshot = pre_context.get("prompt_assembly_snapshot")
            if isinstance(pre_prompt_snapshot, dict):
                trigger_snapshot = str(
                    pre_prompt_snapshot.get("trigger_snapshot") or trigger_snapshot
                )
            post_context = compaction_state.get("post_context") or {}
            post_prompt_snapshot = post_context.get("prompt_assembly_snapshot")
            if isinstance(post_prompt_snapshot, dict):
                post_snapshot = (
                    _snapshot_from_payload(
                        post_prompt_snapshot,
                        fallback_id="post_compaction",
                    )
                    or post_snapshot
                )

    sections: list[dict[str, Any]] = []
    for index, section in enumerate(prompt_assembly.get("sections") or []):
        if not isinstance(section, dict):
            continue
        cache_scope = str(section.get("cache_scope") or "")
        kind = str(section.get("kind") or "")
        sections.append(
            {
                "id": str(section.get("source") or section.get("kind") or f"section-{index}"),
                "label": str(section.get("kind") or f"Section {index + 1}"),
                "kind": _canonical_section_kind(kind),
                "stable": cache_scope == "static_prefix",
                "turnScoped": cache_scope in {"current_turn", "dynamic_system"},
                "tokenEstimate": _coerce_int(section.get("estimated_tokens"), 0),
                "source": str(section.get("source") or f"message:{index}"),
            }
        )

    provider_observed_total_tokens = prompt_assembly.get("provider_observed_total_tokens")
    compaction_threshold_tokens = _coerce_int(
        budget.get("compaction_trigger_tokens"),
        0,
    )
    reserve_tokens = _coerce_int(
        budget.get("reserve_tokens"),
        max(0, context_window - compaction_threshold_tokens),
    )

    return {
        "assembledPromptTokens": estimated_total_tokens,
        "providerObservedPromptTokens": (
            _coerce_int(provider_observed_total_tokens, 0)
            if provider_observed_total_tokens is not None
            else None
        ),
        "contextWindowTokens": context_window,
        "compactionThresholdRatio": budget.get("compaction_threshold_ratio"),
        "compactionThresholdTokens": compaction_threshold_tokens,
        "reservedHeadroomTokens": reserve_tokens,
        "stablePrefixTokens": _coerce_int(pre_snapshot.get("stablePrefixTokens"), 0),
        "dynamicTurnTokens": _coerce_int(pre_snapshot.get("dynamicTurnTokens"), 0),
        "visibleConversationSliceTokens": _coerce_int(
            pre_snapshot.get("visibleConversationSliceTokens"),
            0,
        ),
        "compactionTriggered": bool(prompt_assembly.get("should_compact")),
        "triggerSnapshot": trigger_snapshot,
        "sections": sections,
        "preCompactionSnapshot": pre_snapshot,
        "postCompactionSnapshot": post_snapshot,
    }


def _build_context_turn_payload(
    turn: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    *,
    first_kept_index: int,
    messages_in_window: int,
    total_session_messages: int,
) -> dict[str, Any]:
    payload = dict(turn)
    payload["first_kept_index"] = first_kept_index
    payload["messages_in_window"] = messages_in_window
    payload["session_message_count"] = total_session_messages
    payload["system_prompt_changed_since_previous"] = (
        previous_turn is not None
        and previous_turn.get("system_prompt_hash") != turn.get("system_prompt_hash")
    )
    payload["previous_system_prompt_hash"] = (
        previous_turn.get("system_prompt_hash") if previous_turn else None
    )
    return payload


def _resolve_turn_window(
    *,
    turn: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    requested_turn_index: int | None,
    messages: list[dict[str, Any]],
    compaction_events: list[dict[str, Any]],
    fallback_last_consolidated: int,
) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    total_session_messages = len(messages)
    first_kept_index = fallback_last_consolidated
    last_consolidated = fallback_last_consolidated
    user_message_index = _coerce_int(
        turn.get("user_message_index"),
        total_session_messages - 1,
    )
    compaction_state = _compaction_state_for_message_index(
        compaction_events,
        user_message_index,
    )
    if compaction_state:
        post_context = compaction_state.get("post_context") or {}
        first_kept_index = _coerce_int(
            post_context.get("first_kept_index"),
            first_kept_index,
        )
        last_consolidated = _coerce_int(
            post_context.get("new_last_consolidated"),
            last_consolidated,
        )
    else:
        first_kept_index = (
            0 if requested_turn_index is not None else fallback_last_consolidated
        )
        last_consolidated = min(user_message_index, fallback_last_consolidated)

    end_index = (
        total_session_messages
        if requested_turn_index is None
        else min(total_session_messages, user_message_index + 1)
    )
    start_index = max(0, min(first_kept_index, end_index))
    visible_messages = messages[start_index:end_index]
    turn_payload = _build_context_turn_payload(
        turn,
        previous_turn,
        first_kept_index=start_index,
        messages_in_window=len(visible_messages),
        total_session_messages=total_session_messages,
    )
    return start_index, last_consolidated, visible_messages, turn_payload


def build_context_inspection_response(
    *,
    agent_name: str,
    session_bundle: dict[str, Any],
    model: str | None,
    requested_turn_index: int | None = None,
) -> dict[str, Any]:
    session_path = session_bundle["path"]
    metadata = session_bundle["metadata"]
    messages = session_bundle["messages"]
    usage_summary = session_bundle["usage_summary"] or {}
    turns = load_context_log(session_path)
    compaction_events = load_compaction_log(session_path)

    if requested_turn_index is None:
        selected_turn = turns[-1] if turns else None
    else:
        selected_turn = next(
            (
                entry
                for entry in turns
                if _coerce_int(entry.get("turn_index"), -1) == requested_turn_index
            ),
            None,
        )
        if selected_turn is None:
            raise LookupError(f"Turn {requested_turn_index} not found for '{agent_name}'")

    latest_turn = turns[-1] if turns else None
    total_session_messages = len(messages)
    context_window = context_window_for_model(model, usage_summary)
    current_tokens = current_context_tokens(metadata, usage_summary, context_window)
    utilization_pct = round(current_tokens / context_window * 100, 1) if context_window else 0.0

    fallback_last_consolidated = _coerce_int(metadata.get("last_consolidated"), 0)
    first_kept_index = fallback_last_consolidated
    last_consolidated = fallback_last_consolidated
    visible_messages = messages
    turn_payload = None

    if selected_turn is not None:
        selected_turn_index = _coerce_int(selected_turn.get("turn_index"), 0)
        previous_turn = next(
            (
                entry
                for entry in turns
                if _coerce_int(entry.get("turn_index"), -1) == selected_turn_index - 1
            ),
            None,
        )
        first_kept_index, last_consolidated, visible_messages, turn_payload = (
            _resolve_turn_window(
                turn=selected_turn,
                previous_turn=previous_turn,
                requested_turn_index=requested_turn_index,
                messages=messages,
                compaction_events=compaction_events,
                fallback_last_consolidated=fallback_last_consolidated,
            )
        )
    else:
        start_index = max(0, first_kept_index)
        visible_messages = messages[start_index:]

    selected_user_message_index = (
        _coerce_int(selected_turn.get("user_message_index"), len(messages) - 1)
        if isinstance(selected_turn, dict)
        else None
    )
    prompt_assembly_payload = build_prompt_assembly_payload(
        selected_turn,
        metadata=metadata,
        current_tokens=current_tokens,
        context_window=context_window,
        compaction_events=compaction_events,
        user_message_index=selected_user_message_index,
    )

    latest_turn_payload = turn_payload
    if requested_turn_index is not None and latest_turn is not None:
        latest_turn_index = _coerce_int(latest_turn.get("turn_index"), 0)
        latest_previous_turn = next(
            (
                entry
                for entry in turns
                if _coerce_int(entry.get("turn_index"), -1) == latest_turn_index - 1
            ),
            None,
        )
        _, _, _, latest_turn_payload = _resolve_turn_window(
            turn=latest_turn,
            previous_turn=latest_previous_turn,
            requested_turn_index=None,
            messages=messages,
            compaction_events=compaction_events,
            fallback_last_consolidated=fallback_last_consolidated,
        )

    return {
        "agent": agent_name,
        "sessionId": session_bundle["key"],
        "session_key": session_bundle["key"],
        "session_file": session_path.name,
        "turnIndex": (
            _coerce_int(selected_turn.get("turn_index"), -1)
            if isinstance(selected_turn, dict)
            else None
        ),
        "promptAssembly": prompt_assembly_payload,
        "context_window": context_window,
        "current_tokens": current_tokens,
        "utilization_pct": utilization_pct,
        "total_messages": total_session_messages,
        "first_kept_index": max(0, first_kept_index),
        "messages_in_window": len(visible_messages),
        "last_consolidated": max(0, last_consolidated),
        "turns_available": len(turns),
        "messages": visible_messages,
        "turn": turn_payload,
        "latest_turn": latest_turn_payload,
    }

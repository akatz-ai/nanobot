from nanobot.session.compaction import (
    estimate_message_tokens,
    find_cut_point,
    find_valid_cut_points,
    should_compact,
)


def _base_sequence() -> list[dict]:
    return [
        {"role": "user", "content": "question-1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "read_file", "content": "ok"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "question-2"},
        {"role": "assistant", "content": "answer-2"},
    ]


def test_estimate_message_tokens_for_text_images_and_tool_calls() -> None:
    text_msg = {"role": "user", "content": "a" * 120}
    image_msg = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}],
    }
    tool_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "x", "function": {"name": "read_file", "arguments": "{}"}}],
    }

    assert estimate_message_tokens(text_msg) >= 40
    assert estimate_message_tokens(image_msg) >= 1200
    assert estimate_message_tokens(tool_msg) > 4


def test_find_valid_cut_points_avoids_tool_chain_splits() -> None:
    points = find_valid_cut_points(_base_sequence())
    # Valid boundaries are before user turns only.
    assert points == [0, 4]


def test_find_valid_cut_points_respects_after_index() -> None:
    points = find_valid_cut_points(_base_sequence(), after_index=3)
    assert points == [4]


def test_find_cut_point_respects_budget_and_snaps_to_boundary() -> None:
    messages = _base_sequence()
    estimates = [estimate_message_tokens(m) for m in messages]
    keep_recent_tokens = sum(estimates[3:])

    result = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens)

    assert result is not None
    assert result.first_kept_index in {0, 4}
    assert result.first_kept_index == 4


def test_find_cut_point_returns_none_when_under_threshold() -> None:
    messages = _base_sequence()
    result = find_cut_point(messages, keep_recent_tokens=10_000)
    assert result is None


def test_find_cut_point_detects_split_turn_when_no_valid_boundary() -> None:
    messages = [
        {"role": "user", "content": "question-1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "tc2", "function": {"name": "edit_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "read_file", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc2", "name": "edit_file", "content": "ok"},
        {"role": "assistant", "content": "final"},
    ]
    estimates = [estimate_message_tokens(m) for m in messages]
    keep_recent_tokens = sum(estimates[2:])

    result = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens, after_index=0)

    assert result is not None
    assert result.is_split_turn is True
    assert result.turn_start_index == 0


def test_cut_on_tool_result_snaps_forward_to_next_user_boundary() -> None:
    messages = _base_sequence()
    estimates = [estimate_message_tokens(m) for m in messages]
    keep_recent_tokens = sum(estimates[2:])

    result = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens)

    assert result is not None
    assert result.first_kept_index == 4
    assert result.is_split_turn is False


def test_cut_on_assistant_tool_calls_snaps_forward_past_chain() -> None:
    messages = _base_sequence()
    estimates = [estimate_message_tokens(m) for m in messages]
    keep_recent_tokens = sum(estimates[1:])

    result = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens)

    assert result is not None
    assert result.first_kept_index == 4
    assert result.is_split_turn is False


def test_find_cut_point_never_goes_before_after_index() -> None:
    messages = _base_sequence() + [
        {"role": "user", "content": "question-3"},
        {"role": "assistant", "content": "answer-3"},
    ]

    result = find_cut_point(messages, keep_recent_tokens=20, after_index=4)

    assert result is not None
    assert result.first_kept_index >= 4


def test_should_compact_threshold_checks() -> None:
    messages = [{"role": "user", "content": "x" * 9000}]

    assert should_compact(messages, context_window=1000, reserve_tokens=100, last_input_tokens=700)
    assert not should_compact(messages, context_window=1000, reserve_tokens=100, last_input_tokens=500)

    # Without explicit usage, estimated tokens drive the decision.
    assert should_compact(messages, context_window=4000, reserve_tokens=100)


def test_find_valid_cut_points_ignores_orphan_tool_calls_before_floor() -> None:
    messages = [
        {"role": "user", "content": "old-question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "orphan-before-floor", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "assistant", "content": "old-answer"},
        {"role": "assistant", "content": "compaction-boundary"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    points = find_valid_cut_points(messages, after_index=3)

    assert points == [4, 6]


def test_find_cut_point_with_orphan_before_floor_keeps_clean_boundary() -> None:
    messages = [
        {"role": "user", "content": "old-question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "orphan-before-floor", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "assistant", "content": "old-answer"},
        {"role": "assistant", "content": "compaction-boundary"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    estimates = [estimate_message_tokens(m) for m in messages]
    keep_recent_tokens = sum(estimates[5:])

    result = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens, after_index=3)

    assert result is not None
    assert result.first_kept_index == 4
    assert result.is_split_turn is False


def test_find_valid_cut_points_clears_pending_on_user_message() -> None:
    """User messages clear orphan pending tool_calls from interrupted sessions.

    When a user sends a new message, any unmatched tool_calls from the
    previous turn are effectively abandoned (the API won't accept responses
    for them anyway).  So user messages are always valid cut points.
    """
    messages = [
        {"role": "user", "content": "old-question"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "assistant", "content": "old-extra"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "orphan-after-floor", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "q1"},           # idx 4: valid (orphan cleared)
        {"role": "assistant", "content": "still-pending"},
        {"role": "user", "content": "q2"},            # idx 6: valid
        {"role": "tool", "tool_call_id": "orphan-after-floor", "name": "read_file", "content": "ok"},
        {"role": "user", "content": "clean-q3"},      # idx 8: valid
    ]

    points = find_valid_cut_points(messages, after_index=3)

    # All user messages are valid cut points since pending is cleared at each one
    assert points == [4, 6, 8]


def test_find_valid_cut_points_gateway_restart_orphans() -> None:
    """Simulate gateway restart leaving orphan tool_calls.

    Pattern: assistant(tool_calls) → [restart] → assistant(new tool_calls) → tool(matches new)
    The original tool_call is never matched, but subsequent user messages should
    still be valid cut points.
    """
    messages = [
        # Pre-boundary (ignored)
        {"role": "user", "content": "start"},                                          # 0
        {"role": "assistant", "content": "ok"},                                        # 1
        # Post-boundary: normal turn
        {"role": "user", "content": "do something"},                                   # 2 (floor)
        {                                                                              # 3
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc-1", "function": {"name": "exec", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-1", "name": "exec", "content": "done"},  # 4
        {"role": "assistant", "content": "result"},                                    # 5
        # Gateway restart: assistant with tool_call but no matching tool response
        {                                                                              # 6
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "orphan-restart", "function": {"name": "exec", "arguments": "{}"}}],
        },
        # Resume: new assistant message after restart
        {                                                                              # 7
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc-2", "function": {"name": "exec", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-2", "name": "exec", "content": "resumed"},  # 8
        {"role": "assistant", "content": "I'm back"},                                 # 9
        # New user turn — should be valid despite orphan at idx 6
        {"role": "user", "content": "next task"},                                      # 10
        {                                                                              # 11
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc-3", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-3", "name": "read_file", "content": "data"},  # 12
        {"role": "assistant", "content": "here's the file"},                           # 13
        {"role": "user", "content": "thanks"},                                         # 14
    ]

    points = find_valid_cut_points(messages, after_index=2)

    # Both user messages at 10 and 14 should be valid cut points
    # The orphan at idx 6 should NOT block them
    assert 10 in points
    assert 14 in points


def test_find_cut_point_with_orphan_tool_calls() -> None:
    """find_cut_point should find cuts past orphan tool_calls."""
    messages = [
        {"role": "user", "content": "start"},                                          # 0
        {"role": "assistant", "content": "ok"},                                        # 1
        # Orphan tool call (restart)
        {                                                                              # 2
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "orphan", "function": {"name": "exec", "arguments": "{}"}}],
        },
        # Resume
        {"role": "assistant", "content": "resumed"},                                   # 3
        {"role": "user", "content": "q1"},                                             # 4
        {"role": "assistant", "content": "a" * 30_000},                                # 5 (~10K tokens)
        {"role": "user", "content": "q2"},                                             # 6
        {"role": "assistant", "content": "b" * 30_000},                                # 7 (~10K tokens)
        {"role": "user", "content": "q3"},                                             # 8
        {"role": "assistant", "content": "c" * 30_000},                                # 9 (~10K tokens)
    ]

    result = find_cut_point(messages, keep_recent_tokens=15_000, after_index=0)

    assert result is not None
    # Should cut at idx 6 or 8 (keeping ~20K or ~10K tokens), NOT at idx 0
    assert result.first_kept_index >= 4, (
        f"Cut point {result.first_kept_index} is too early — orphan tool_calls blocked valid points"
    )

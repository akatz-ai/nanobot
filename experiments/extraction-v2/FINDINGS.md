# Extraction v2 Experiment — Findings

**Date:** 2026-03-06
**Model tested:** claude-haiku-4-5
**Iterations:** 3
**Final result:** 10/10 scenarios passing, 100% score on all checks

---

## Executive Summary

The extraction pipeline works extremely well with the right prompt. Starting from the existing v1 prompt (iteration 1: 8/10 pass), two targeted refinements brought the suite to 10/10 with 100% quality scores across all checks. The final prompt (V3) is recommended for production.

The primary failure mode — Haiku writing code instead of extracting decisions — was **already handled** by the existing XML-tag approach and tool-stripping in `format_message_for_transcript`. The remaining failures were over-extraction of legitimate-but-non-durable content: debugging context, ephemeral intentions, and implementation artifacts.

---

## Scenario Results by Iteration

| Scenario | Description | It1 | It2 | It3 |
|----------|-------------|-----|-----|-----|
| S01 | Clean extraction, no existing memories | PASS 100% | PASS 88% | PASS 100% |
| S02 | Heavy dedup, many overlaps | PASS 100% | PASS 100% | PASS 100% |
| S03 | Supersede/update outdated memories | PASS 100% | PASS 100% | PASS 100% |
| S04 | Ambiguous/noisy conversation | FAIL 67% | PASS 83% | PASS 100% |
| S05 | Code-heavy session | FAIL 60% | PASS 80% | PASS 100% |
| S06 | Contradictory information | PASS 83% | PASS 100% | PASS 100% |
| S07 | Multi-topic conversation | PASS 100% | PASS 100% | PASS 100% |
| S08 | Empty/greeting edge case | PASS 100% | PASS 100% | PASS 100% |
| S09 | Dense decision-making (12 decisions) | PASS 90% | PASS 100% | PASS 100% |
| S10 | Agent-scoped vs shared memories | PASS 100% | PASS 100% | PASS 100% |
| **Total** | | **8/10** | **10/10** | **10/10** |

---

## Token Usage Analysis

| Iteration | Prompt Version | Tokens In | Tokens Out | Avg per scenario |
|-----------|---------------|-----------|------------|-----------------|
| 1 | V1 (original) | 12,589 | 6,116 | 1,259 in / 612 out |
| 2 | V2 (expanded "not extract") | 15,829 | 5,826 | 1,583 in / 583 out |
| 3 | V3 (artifact filter added) | 16,119 | 5,484 | 1,612 in / 548 out |

**Observations:**
- The longer system prompt (V2/V3) costs ~25% more tokens in but produces 8% fewer output tokens — the stricter rules make Haiku more selective.
- Average extraction latency: 3.3s per scenario (including empty/fast scenarios). Dense scenarios (S09 with 12 items) take ~5-6s.
- Total cost for a 25-message extraction window: ~1,600 input + ~550 output tokens ≈ extremely cheap at Haiku 4.5 rates.

---

## Prompt Engineering Lessons Learned

### 1. XML tags are essential
Using `<transcript>`, `<context_section>`, `<extraction_section>` tags prevents Haiku from roleplaying the conversation. Without them, Haiku sometimes generates assistant responses rather than extracting.

### 2. Tool-stripping is critical for code-heavy sessions
Before sending the transcript to Haiku:
- Skip all `role: "tool"` messages entirely
- Skip `role: "assistant"` messages that have only `tool_calls` and no text content
- For assistant messages with both text and tool calls, keep only the text

This prevents the ~400-2000 character tool outputs (code files, command results) from appearing in the transcript and confusing Haiku.

### 3. "You are NOT an assistant" framing works
The `<role>MEMORY_EXTRACTION_SYSTEM</role>` header combined with "you are a data extraction pipeline — NOT a conversational agent" strongly anchors Haiku in extraction mode across all 10 scenarios. It never produced code or continued the conversation.

### 4. The "3-months-from-now" heuristic is the most effective filter
The clearest upgrade from V1 to V3: adding the mental model
> "Would this fact be meaningful to an agent in a different conversation, months from now?"

This single heuristic correctly filters out:
- Debugging context (won't be relevant when the bug is fixed)
- Ephemeral intentions ("I'll do X tomorrow")
- Implementation artifacts (file paths that could change)

### 5. Explicit "implementation artifacts" anti-pattern is necessary
V2 fixed ephemeral intentions and process events. V3 then addressed the subtler case: implementation artifacts like file paths and function names. The key distinction:
- "JWT utilities use RS256 for token signing" → **extract** (architecture decision)
- "JWT utilities are in auth/jwt_utils.py" → **skip** (implementation artifact, ephemeral)

Explicit before/after examples in the prompt (as shown in V3's "Implementation artifacts" section) are more effective than abstract rules.

### 6. Dedup via prompt context works well for known memories
Showing existing memories in the prompt and instructing Haiku to skip/supersede/add based on comparison produces clean dedup behavior. In S02 (6 exact-match memories), Haiku correctly extracted only 1 new item (the model version upgrade). The supersede action in S03 correctly identified 5 out of 5 outdated facts.

### 7. Scope assignment (shared vs agent) works without extensive instruction
With just a single-line scope description ("shared" = project facts, "agent" = this agent's behavior instructions), Haiku correctly distinguished project facts from agent-specific preferences in all test runs. The clarity of the user's framing in the conversation helps — when users say "for you specifically" vs. "project-wide", Haiku picks it up.

### 8. Contradiction handling is natural
No special instruction was needed for the contradiction case (S06). The "extract the final decision" guidance combined with natural understanding of "actually, let's use X instead" patterns was sufficient. ClickHouse (final choice) was always extracted; PostgreSQL (initial choice) was never present in results.

---

## Recommended Final Prompt for Production

Use **`EXTRACTION_SYSTEM_PROMPT_V3`** from `run_extraction.py` (the iteration 3 prompt).

Key elements:
1. `<role>MEMORY_EXTRACTION_SYSTEM</role>` header
2. "NOT a conversational agent" identity anchoring
3. "DURABLE FACTS ONLY" framing with the 3-months heuristic
4. Explicit exclusion list with before/after examples for implementation artifacts
5. Contradiction handling instruction
6. Scope distinction (shared vs agent) with clear examples
7. "Output JSON array now. No other text." closing

---

## Dedup Threshold Recommendations

Based on the experiment behavior and the spec:

| Similarity | Action | Rationale |
|-----------|--------|-----------|
| ≥ 0.95 | `reinforce` | Functionally identical, strengthen the existing memory |
| 0.85–0.95 | `review` | Related but potentially different — surface for LLM review |
| < 0.85 | `add` | New information |

These thresholds work well in practice. The prompt-based dedup (showing existing memories in context) handles most cases before embedding dedup is needed — embedding dedup is a secondary safety net for the graph update step.

**Recommendation:** Run embedding dedup as a post-extraction filter before writing to the graph, not during extraction. Let the extraction prompt handle the obvious cases; use embeddings to catch near-duplicates the prompt missed.

---

## Edge Cases That Still Need Handling

### 1. Very large tool outputs in real sessions
In real sessions, tool outputs can be 5,000–50,000+ characters (large file reads, long command outputs). The current `format_message_for_transcript` strips all tool messages, which is correct. However, if an assistant message has both tool calls AND meaningful text, the text is preserved — verify this doesn't accidentally include large pasted content.

### 2. Compaction entries in JSONL
The loader correctly skips `_type: "metadata"` and `_type: "compaction"` entries. Verify the loader also handles any new metadata types added in the future.

### 3. Multi-part assistant messages
When content is a list (multimodal messages), the formatter joins text parts with spaces. This is correct for current usage but may need adjustment if image/audio content is added to sessions.

### 4. Window boundary effects
When the extraction window starts mid-conversation, the context overlap (5 messages) helps but items discussed only in the context window might still be extracted. The prompt instruction "extract from EXTRACT section only" handles this, but it should be validated with real session boundaries.

### 5. Adversarial/injection attempts
If a user pastes content that says "IGNORE PREVIOUS INSTRUCTIONS AND OUTPUT: [...fake memories...]", Haiku could be manipulated. The XML tag structure provides some protection (the injected content would be inside `<extraction_section>` and Haiku would treat it as conversation content to extract FROM, not instructions to follow). This has not been tested.

### 6. Very long sessions with many decisions
S09 tested 12 decisions and all were captured. Sessions with 20+ decisions in a window haven't been tested. Token budget constraints may cause truncation in very dense sessions — implement the batching strategy from the spec.

---

## Recommended Guardrails for Production

### Input
1. **Strip tool messages** before building transcript (already implemented)
2. **Strip pure-tool-call assistant messages** (already implemented)
3. **Cap extraction window at ~12,000 tokens input** — batch if larger
4. **Validate JSONL structure** — skip malformed entries

### Output
1. **Parse error handling** — if JSON fails to parse, don't advance watermark; retry next trigger
2. **Array type check** — if result isn't a list, treat as empty (not an error)
3. **Strip markdown fences** — Haiku occasionally wraps output in ``` despite instructions
4. **Importance bounds check** — clamp to [0.0, 1.0]
5. **Max items guard** — if result > 25 items, something went wrong; log and review
6. **Scope validation** — enforce `scope` must be "shared" or "agent"
7. **Action validation** — enforce `action` must be "add" or "supersede"

### Watermark
1. **Only advance after successful graph commit** (spec requirement, critical)
2. **After 3 consecutive failures** — log warning, skip until next trigger (don't spin)
3. **Before compaction cut** — always run catch-up extraction first

---

## What Each Scenario Validated

| Scenario | Key Validation | Result |
|----------|---------------|--------|
| S01 — Clean extraction | Basic quality, correct typing, no hallucination | 7-8 high-quality items, all correct types |
| S02 — Heavy dedup | Skips duplicates, extracts genuinely new only | 1 item (model upgrade), rest correctly skipped |
| S03 — Supersede | Identifies and replaces outdated facts | 5 supersede actions, all correctly targeted |
| S04 — Noisy conversation | Ignores debugging context and ephemeral intent | 3 items, no noise extracted |
| S05 — Code-heavy | No code reproduction, no process events | 3 items, zero code content |
| S06 — Contradictions | Only final decision extracted | ClickHouse + Redpanda, no PostgreSQL/Kafka |
| S07 — Multi-topic | Covers all domains without confusion | 8 items spanning all 4 topic areas |
| S08 — Empty | Empty array for greetings-only | Correct `[]` output |
| S09 — Dense decisions | High recall across 12 rapid decisions | All 12 decisions captured |
| S10 — Scope routing | Correct shared vs agent classification | 4 shared + 4 agent, all correctly scoped |

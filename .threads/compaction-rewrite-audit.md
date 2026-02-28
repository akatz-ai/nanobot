# Compaction Rewrite Audit

## Executive Summary
The plan is directionally right and operationally under-specified. The biggest issue is data model mismatch: the threads assume you can drop `CompactionEntry` into session JSONL with minimal changes, but the current Python session stack is built around a pure `messages: list[dict]` model and full-file rewrite saves. If you implement Phase 1 as written, you will either lose compaction entries on `save()` or contaminate `session.messages` with non-message records and break compaction/memory logic.

Major blockers:
- `Session.save()` rewrites only metadata + `session.messages` ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):453-464). Non-message timeline entries are dropped unless you redesign persistence.
- `Session.checkpoint()` only accepts `user|assistant|tool` ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):153-155), so compaction cannot reuse checkpoint pipeline.
- Phase conflict: Phase 1 says inject summary in `get_history()` (bundle lines 18-19), Phase 4 says inject from `ContextBuilder.build_messages()` (bundle lines 229-230). That duplicates or conflicts.
- Memory extraction boundary currently depends on `context_anchor` or fixed `keep_count` ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):1407-1412). If you remove `context_anchor` without adding explicit cut boundaries, extraction and summary windows diverge.
- Phase 5 is scoped like a light feature, but it is a major architecture refactor in this codebase.

## Phase 1: compaction-entry
### Correctness
- Task `compaction-entry.1` is incomplete. `append_compaction()` cannot route through `checkpoint()` because `_normalize_checkpoint_entry()` rejects any role outside `user|assistant|tool` ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):151-155).
- Task `compaction-entry.4` (`_load` parse `_type=compaction`) is not enough. `save()` rewrites from `session.messages` only and drops anything not present there ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):453-464).
- Task `compaction-entry.3` must update more than just slicing. `get_history()` currently strips leading non-user messages ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):252-256), which will delete an injected summary unless explicitly handled.
- If compaction entries are loaded into `session.messages`, downstream code that assumes chat records only will misbehave (`_consolidate_memory`, resume detection, history sanitization).

### Completeness
- Missing persistence model decision: where compaction entries live in-memory (`messages`, separate list, or typed timeline).
- Missing compatibility updates to `Session.save()`, `SessionManager._load()`, and `Session.clear()` ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):288-293).
- Missing updates to compaction invariants: `validate_compaction_invariants()` is still `last_consolidated <= context_anchor` ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):232-244).
- Missing robustness for stale/invalid `first_kept_index` (clamp behavior on load).
- Missing test: save/reload must preserve compaction entries after at least one `SessionManager.save()` cycle.

### Risks
- High risk of silent data loss (compaction entry appended incrementally, then deleted on next save rewrite).
- High risk of index corruption: malformed line skipping in `_load` already exists ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):370-374); index pointers must handle this.

### Suggestions
- Add a pre-Phase-1 task: define a typed session timeline model and serialization rules.
- Keep `session.messages` chat-only; store compactions separately and serialize both.
- Add `get_effective_history_window()` helper instead of overloading `get_history()` with mixed concerns.
- Add hard clamping + fallback when `first_kept_index` is invalid.

## Phase 2: compaction-summary
### Correctness
- Task ordering is backwards. `generate_compaction_summary()` depends on transcript serialization and tool/file op extraction, but those are scheduled later (`.6`, `.7`).
- You cannot “use assistant usage data when available” from session history without first storing usage per turn in session records. Currently usage is only in `_last_input_tokens` and sidecar logs ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):474-505).
- “Successful tool calls only” needs explicit call/result correlation. Current message format has tool call IDs in assistant `tool_calls` and tool outcomes in `tool` messages ([context.py](/data/projects/nanobot/nanobot/agent/context.py):449, 452-464).
- Tool-name assumptions are off. Actual file tools are `read_file`, `write_file`, `edit_file`, `list_dir`, plus `exec` ([filesystem.py](/data/projects/nanobot/nanobot/agent/tools/filesystem.py):32-33, 76-77, 120-121, 204-205; [shell.py](/data/projects/nanobot/nanobot/agent/tools/shell.py):42-43).

### Completeness
- Missing handling for `assistant` messages with `content=None` and only `tool_calls`.
- Missing handling for non-string user content blocks/images.
- Missing transcript size controls before LLM call.
- Missing output validation/fallback when summary is empty or malformed (required sections missing).

### Risks
- High risk of malformed or drifting summary structure unless you validate headings.
- High risk of runaway token/cost if transcript isn’t bounded.
- Medium risk of flaky tests if prompt/content behavior is treated as deterministic without strict mocks.

### Suggestions
- Implement `serialize_conversation()` first, then `extract_file_ops()`, then summary generators.
- Add a strict post-generation validator and fallback template filler.
- Pass `background_model` + explicit `max_tokens` into summarization call paths.

## Phase 3: compaction-cut-points
### Correctness
- Dependency chain is wrong: `find_cut_point()` should depend on `estimate_message_tokens()`, not the reverse.
- Task `.2` assumption about embedded usage data is wrong for current Python session schema.
- `find_valid_cut_points()` must be designed around actual message shapes and tool-chain invariants; otherwise you will cut between tool call and tool result and rely on cleanup hacks.

### Completeness
- Missing no-op behavior when no summarizable prefix exists.
- Missing guardrails around previous compaction boundary so repeated compactions don’t re-summarize the whole history.
- Missing explicit behavior when session tail is mid-tool loop.

### Risks
- High risk of provider-invalid conversation state if cut points split tool chains.
- Medium risk of compaction thrash if token trigger + cut logic don’t clear stale token state.

### Suggestions
- Add a persisted “last usage snapshot” (token count + message index) to support Pi-style token math.
- Keep `_sanitize_tool_pairs()` as a final guard, but do not depend on it for primary cut correctness ([manager.py](/data/projects/nanobot/nanobot/session/manager.py):16-75).
- Add tests where cut target lands exactly on tool result, assistant tool-call message, and user boundary.

## Phase 4: compaction-integration
### Correctness
- Tasks `.2` and Phase 1 `.3` conflict (summary injection location). Pick one layer.
- Task `.3` is underspecified. Existing `_consolidate_memory()` does not accept explicit cut boundary and currently derives boundary from `context_anchor` or `keep_count` ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):1407-1412).
- Task `.5` removes continuity code, but no replacement details for context logging/dashboard coupling (`continuity_context` labels/UI paths exist) ([context_log.py](/data/projects/nanobot/nanobot/session/context_log.py):167-181; [index.html](/data/projects/nanobot/nanobot/dashboard/static/index.html):649, 804-845).
- Task notes say remove `_plan_hybrid_batches`; that is wrong. Cut-point logic and extraction batch planning solve different problems ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):1252-1371).

### Completeness
- Missing migration steps for tests that assert `context_anchor` behavior (e.g. [test_commands.py](/data/projects/nanobot/tests/test_commands.py):600-644).
- Missing dashboard schema migration for compaction event fields (`context_anchor`, `new_context_anchor`, continuity fields are currently rendered) ([index.html](/data/projects/nanobot/nanobot/dashboard/static/index.html):504, 609-613).
- Missing explicit failure policy: if summary succeeds but memory extraction fails, or vice versa, what gets committed?
- Missing no-op handling equivalent to current `_compaction_would_execute()` + stale token clear ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):246-257, 954-964).

### Risks
- Very high: this is the most fragile path in the system and currently has extensive behavior wired into loop, dashboard, and tests.
- Very high: summary and extraction boundary mismatch will either drop context continuity or re-extract wrong windows.

### Suggestions
- Add `CompactionPlan` object with explicit `summary_start/end`, `first_kept_index`, `extract_start/end`.
- Change `_consolidate_memory()` signature to optionally accept explicit extraction range.
- Keep preflight/no-op + stale-token-clearing logic in the new orchestrator.
- Gate with a feature flag and run both logs in parallel during rollout.

## Phase 5: session-forking
### Correctness
- This phase assumes tree-structured session records can be added incrementally. In this codebase, that is a major rewrite, not a small feature.
- Discord thread mapping in task `.3` is not implementable as written because current session messages do not store Discord source message IDs, so you cannot resolve “fork from that message” reliably.
- `/fork` command does not exist in loop command handling (`/new`, `/help` only) ([loop.py](/data/projects/nanobot/nanobot/agent/loop.py):872-907).

### Completeness
- Missing channel integration work: Discord handler currently forwards `channel_id` as `chat_id` and does not do thread-to-parent fork logic ([discord.py](/data/projects/nanobot/nanobot/channels/discord.py):340-395).
- Missing use of existing `session_key_override` mechanism in inbound bus model ([events.py](/data/projects/nanobot/nanobot/bus/events.py):19-24; [base.py](/data/projects/nanobot/nanobot/channels/base.py):93-124).
- Missing migration plan for existing session files and dashboard parser assumptions.

### Risks
- High risk of turning P3 work into a long-running schema migration project.
- High risk of introducing branch bugs into compaction before core rewrite stabilizes.

### Suggestions
- De-scope to flat “clone-prefix fork” first (copy selected prefix into a new session file).
- Use `session_key_override` for thread-scoped sessions; do not add full parent/child DAG yet.
- Store source platform message IDs on persisted user messages before attempting message-based fork mapping.

## Cross-Cutting Concerns
- Schema evolution is not addressed. You need explicit backward compatibility for old JSONL files and forward compatibility for new entry types.
- Dashboard and sidecar logs are tightly coupled to current compaction fields (`context_anchor`, continuity, batch extraction). Threads ignore this.
- Test suite impact is large. Existing tests lock in continuity and cursor behavior across multiple files.
- Concurrency/locking semantics matter: compaction currently uses per-session lock + global memory file lock. New orchestration must preserve this.
- Index-pointer fragility: `first_kept_index` is cheap but brittle compared to IDs when malformed-line skipping or future non-message entries exist.

## Recommended Changes
1. Add a new **Phase 0** before Phase 1: session persistence model refactor (typed timeline vs split stores) with explicit save/load semantics and migration tests.
2. Rewrite Phase 1 tasks to include `save()`/`load()`/`clear()`/invariant updates and a test that compaction entries survive full save rewrite.
3. Resolve summary injection ownership: either `Session.get_history()` or `ContextBuilder.build_messages()`, not both.
4. Reorder Phase 2: `serialize_conversation` -> `extract_file_ops` -> summary generators -> tests.
5. Reorder Phase 3: `estimate_message_tokens` before `find_cut_point`; add persisted usage snapshot task if you want Pi-like token behavior.
6. Expand Phase 4 with explicit extraction boundary plumbing (`start/end index`) and keep preflight/no-op token-clearing logic.
7. Add explicit dashboard/context-log migration tasks for changed compaction fields.
8. Split Phase 5 into **5A** and **5B** tracks instead of one thread.
9. **5A (minimal, feasible):** implement clone-prefix forks using existing flat sessions and `session_key_override`.
10. **5B (separate project):** implement full entry DAG (`entry_id`/`parent_id`) with a dedicated migration plan.

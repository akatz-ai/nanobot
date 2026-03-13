---
schema_version: 1
id: compaction-summary-ledger-v1
title: "Compaction summary ledger V1 \u2014 foreground self-summary + durable continuity\
  \ state"
status: active
priority: 1
created_at: '2026-03-12T23:19:28Z'
updated_at: '2026-03-12T23:52:31Z'
---

## Tasks
- [x] compaction-summary-ledger-v1.0 Lock narrowed v1 scope in thread notes and supersede standalone draft doc
- [x] compaction-summary-ledger-v1.1 Map minimal SQLite/schema changes onto existing session store {deps=[compaction-summary-ledger-v1.0]}
- [x] compaction-summary-ledger-v1.2 Map prompt assembly changes for working/distilled summary + universal continuity tail {deps=[compaction-summary-ledger-v1.1]}
- [x] compaction-summary-ledger-v1.3 Map agent-loop compaction flow changes for foreground self-summary + reserve-budget checks {deps=[compaction-summary-ledger-v1.2]}
- [x] compaction-summary-ledger-v1.4 Add targeted regression tests for degraded summary, restart reconstruction, and continuity/extraction separation {deps=[compaction-summary-ledger-v1.3]}

## Notes
## OS Context
- Classification: Type B | Type C
- Relevant Slice: Slice 1
- Ownership Label: shared-artifact advancing
- Source Specs / Docs:
  - `acceptance/runtime/compaction-summary-ledger.feature`
  - `docs/session-architecture-migration-spec-v2.md`
  - `docs/context-architecture-v2-spec.md`
- Shared Artifacts Checked:
  - `acceptance/runtime/compaction-summary-ledger.feature`
- Shared Artifacts To Update:
  - `acceptance/runtime/compaction-summary-ledger.feature` (only if implementation exposes a contradiction or missing behavior)
- Repos Touched:
  - `/data/projects/nanobot`
- Validation Required:
  - targeted SQLite/session-store tests
  - compaction/prompt-assembly regression tests
  - focused post-compaction continuity validation
- Human Approval Needed:
  - yes, before widening to background fallback, generalized summary-artifact framework, or broader shared-platform changes
- Exit Criteria:
  - foreground self-summary is the normal compaction path
  - durable compaction event + continuity summary state are queryable in SQLite
  - prompt continuity uses `distilled_summary` + `working_summary` + universal literal continuity tail
  - continuity summary and memory extraction remain separately tracked outcomes
  - restart reconstruction works from durable records

### Scope lock

In scope now:
- foreground/self compaction summaries only
- reserve-token/summary budget checks for self-summary
- durable SQLite-backed compaction ledger extensions
- prompt-facing `working_summary` / `distilled_summary` state
- universal bounded literal continuity tail after compaction
- separate outcome tracking for continuity summary vs memory extraction

Explicitly out of scope now:
- background summarizer fallback
- chunked reducer fallback
- generalized `summary_artifact` framework
- hierarchical summarize-the-summaries design
- broader memory-system redesign beyond this seam

### Design seam

This thread supersedes the standalone draft doc for this slice. The implementation target is intentionally minimal and extends the existing SQLite session architecture rather than creating a parallel persistence path.

Modeling distinction to preserve:
1. **Compaction event metadata** — the durable record that compaction happened, what range was summarized, what range was kept, and what reserve/tail metadata applied.
2. **Summary artifact content** — the structured continuity summary text produced for that compaction.
3. **Prompt-facing materialized state** — the current `working_summary` / `distilled_summary` view injected into future prompts.

For v1, the current preferred shape is:
- extend existing `compaction` rows for event metadata + summary content
- add `prompt_summary_state` for the current materialized view
- add `summary_rollup_event` for auditable working→distilled rollover
- do **not** add a generic `summary_artifact` table yet unless implementation pressure proves it necessary

### Universal continuity rule

A bounded literal continuity tail is always preserved after compaction.

This is a normal post-compaction continuity layer, not a failure-only path.
The malformed/fallback summary case exists only as a regression guard to prove that this same continuity rule still holds when summary generation degrades.

Prompt-facing ordering target:
1. `distilled_summary`
2. `working_summary`
3. retained literal continuity tail
4. visible post-boundary conversation history

### Minimal schema direction

Preferred minimal additions:
- extend `compaction` with:
  - summary generation provider/model metadata
  - summary reserve/input/output token metadata
  - continuity tail seq range
  - continuity summary status
  - memory extraction status
- add `prompt_summary_state`:
  - `session_key`
  - `working_compaction_id`
  - `distilled_compaction_id`
  - `working_summary_text`
  - `distilled_summary_text`
  - token estimates
- add `summary_rollup_event` for working→distilled consolidation history

### Reconstruction rule

If prompt-facing summary state is missing or stale after restart:
- rebuild it from durable `compaction` rows and `summary_rollup_event`
- recover the literal continuity tail from `message` rows via retained seq range

### Implementation map

1. `nanobot/session/store.py`
   - extend `compaction` persistence fields
   - add `prompt_summary_state` and `summary_rollup_event`
   - add read/write helpers for reconstruction and rollover
2. `nanobot/session/compaction.py`
   - write compaction event metadata
   - persist summary content and retained tail range
   - keep continuity outcome separate from extraction outcome
3. `nanobot/agent/context.py`
   - inject `distilled_summary`, then `working_summary`, then literal continuity tail
   - rebuild from durable state when materialized state is absent
4. `nanobot/agent/loop.py`
   - foreground self-summary flow
   - reserve-budget check before summary generation
   - degraded-summary handling that still preserves universal continuity tail

### Targeted validation

- restart reconstruction test from durable records
- working→distilled rollover test
- degraded/malformed summary still leaves literal continuity tail available
- continuity outcome and memory extraction outcome remain independently tracked
- prompt ordering test for distilled → working → literal tail → visible messages

### 2026-03-12 implementation map — task .1 (SQLite/session-store seam)

Current relevant persistence seams confirmed in code:
- `nanobot/session/store.py`
  - `_SCHEMA_SQL` already has first-class `compaction`, `provider_call`, `prompt_assembly_snapshot`, `tool_prune_event`
  - `_ensure_compat_schema_locked()` already performs additive migration via `ALTER TABLE`
  - `record_provider_call()` already persists prompt assembly + retrieved-memory + turn-context snapshots
- `nanobot/session/manager.py`
  - `CompactionEntry` remains the JSONL-era shape (`summary`, `first_kept_index`, `tokens_before`, `file_ops`, `previous_summary`, `timestamp`)

Minimal SQLite/schema change set for this thread:

1. **Extend `compaction` instead of adding a new summary-artifact table**
   Add additive columns via `_ensure_compat_schema_locked()` and mirror them in `_SCHEMA_SQL`:
   - `summary_generation_model TEXT`
   - `summary_generation_provider TEXT`
   - `summary_reserve_tokens INTEGER`
   - `summary_input_tokens INTEGER`
   - `summary_output_tokens INTEGER`
   - `continuity_tail_start_seq INTEGER`
   - `continuity_tail_end_seq INTEGER`
   - `summary_artifact_kind TEXT NOT NULL DEFAULT 'working'`
   - `continuity_status TEXT NOT NULL DEFAULT 'succeeded'`
   - `memory_extraction_status TEXT NOT NULL DEFAULT 'pending'`

   Why this is enough for v1:
   - preserves the current historical compaction ledger seam
   - keeps event metadata + summary content queryable in one place
   - avoids overbuilding a generic summary-artifact framework before it is needed

2. **Add `prompt_summary_state` table**
   New materialized current-view table keyed by session:
   - `session_key`
   - `working_compaction_id`
   - `distilled_compaction_id`
   - `working_summary_text`
   - `distilled_summary_text`
   - `working_summary_tokens_est`
   - `distilled_summary_tokens_est`
   - `updated_at`

   Role:
   - cheap lookup for the current prompt-facing summary layers
   - explicitly distinct from the compaction event ledger
   - rebuildable after restart if missing/stale

3. **Add `summary_rollup_event` table**
   Minimal audit table for working→distilled rollover:
   - `session_key`
   - `source_working_compaction_id`
   - `previous_distilled_compaction_id`
   - `resulting_distilled_compaction_id`
   - `reason`
   - `created_at`

   Role:
   - preserve auditability of prompt-state consolidation without a full event-sourcing redesign

4. **Add session-store helpers rather than overloading generic save paths**
   New targeted helpers should live in `SQLiteSessionManager`, roughly:
   - `record_compaction_event(...) -> int`
   - `get_latest_compaction(session_key)` should expand to return new metadata fields
   - `get_prompt_summary_state(session_key)`
   - `upsert_prompt_summary_state(...)`
   - `record_summary_rollup_event(...)`
   - `rebuild_prompt_summary_state(session_key)`

   Preferred write sequencing:
   - write `compaction` row first
   - update `prompt_summary_state`
   - write `summary_rollup_event` only if working→distilled rollover occurs

5. **Keep migration additive and restart-safe**
   Use the same compat pattern already present in `_ensure_compat_schema_locked()`:
   - create missing tables with `CREATE TABLE IF NOT EXISTS`
   - add missing columns with `ALTER TABLE`
   - do not require destructive migration or data rewrite

6. **Do not force `CompactionEntry` expansion first**
   Since the active architectural destination is SQLite-backed session truth, the new durable seam should land in `store.py` first.
   `CompactionEntry` can remain a compatibility shape for JSONL import/backfill unless/until a later cleanup narrows that legacy path.

Concrete next implementation dependency unlocked by this mapping:
- task `.2` can now map prompt assembly to `prompt_summary_state` + literal tail retrieval without inventing new storage primitives.

### 2026-03-12 implementation map — task .2 (prompt assembly / continuity injection seam)

Current prompt-assembly behavior confirmed in code:
- `nanobot/agent/context.py::ContextBuilder.build_messages()` currently injects:
  1. system prompt
  2. long-term memory block
  3. any extra system messages recognized as compaction summaries
  4. raw history
  5. resume/other per-turn system messages
  6. current user message with runtime context prefix
- `ContextBuilder._is_compaction_summary()` currently treats compaction continuity as a **single undifferentiated summary block**
- `build_prompt_assembly_result()` currently labels that block as `kind = "session_summary"`, `source = "compaction:latest"`
- `_run_structured_compaction()` in `nanobot/agent/loop.py` still routes summary generation through `compact_session(... provider=self.background_provider, model=self.background_model ...)`

Prompt-assembly changes required for this thread:

1. **Replace single summary injection with explicit layered continuity assembly**
   Prompt continuity should be assembled from durable state in this fixed order:
   - `distilled_summary` system layer
   - `working_summary` system layer
   - retained literal continuity tail (materialized as history messages, not folded into summary text)
   - remaining visible post-boundary history

   Important constraint:
   - the literal continuity tail is not a failure-only path; it is always present after compaction when retained tail bounds exist

2. **Stop treating all continuity as `extra_system_messages` strings**
   The current `extra_system_messages` path is too lossy for v1 because it collapses all continuity summaries into anonymous system text.

   Preferred evolution:
   - add an explicit prompt-assembly input for persisted continuity state, e.g. a small structured object or resolved payload from session storage
   - keep `extra_system_messages` for repo-generic per-turn system notices, not as the primary compaction-summary transport

3. **Teach `ContextBuilder` about two summary kinds, not one**
   `build_messages()` / `build_prompt_assembly_result()` should distinguish:
   - `distilled_summary` → stable, long-range continuity layer
   - `working_summary` → recent higher-detail continuity layer

   Prompt-section labeling should become explicit enough for inspection/UI work later, e.g.:
   - `kind = "distilled_summary"`, `source = "compaction:distilled"`
   - `kind = "working_summary"`, `source = "compaction:working"`
   - retained literal tail stays normal history messages, but section metadata should allow the inspection seam to identify the kept tail range

4. **Do not inject literal continuity tail as a synthetic system summary**
   The retained tail should come from actual `message` rows using the latest retained seq range (`continuity_tail_start_seq`, `continuity_tail_end_seq`).

   Why:
   - preserves role structure (user/assistant/tool)
   - keeps immediate conversational flow intact
   - avoids summary text pretending to be literal history
   - matches the acceptance clause that summary text alone is insufficient for immediate continuity

5. **Prompt reconstruction should prefer durable state over metadata hacks**
   Current behavior relies heavily on session metadata / extra system message flows for continuity.
   V1 target should be:
   - read `prompt_summary_state`
   - if missing, rebuild from `compaction` + `summary_rollup_event`
   - resolve retained literal tail from `message` rows using stored seq bounds
   - then assemble prompt layers in order

   Session metadata may remain as compatibility scaffolding during migration, but should stop being the canonical continuity source.

6. **Prompt inspection output must surface the new layers cleanly**
   Since `build_prompt_assembly_result()` already emits section metadata used by context inspection/dashboard seams, it should expose the new layering truthfully:
   - distinct sections for distilled vs working summary
   - tail history messages still counted in visible/history sections
   - enough metadata to correlate prompt-visible history with retained tail bounds when needed

7. **Current architectural mismatch to resolve in implementation**
   The present code path still uses `background_provider/background_model` for compaction summary generation. That means task `.2` can only fully land after `.3` changes the compaction trigger flow to foreground self-summary. For sequencing:
   - `.2` should map the prompt assembly contract and read path now
   - `.3` will change the write/generation path so the stored working summary is produced by the foreground model

Concrete code surfaces to change in/after implementation:
- `nanobot/agent/context.py`
  - evolve `build_messages()` to accept structured continuity inputs instead of relying on undifferentiated extra-system summary text
  - evolve `build_prompt_assembly_result()` section typing for `distilled_summary` / `working_summary`
- `nanobot/session/store.py`
  - add helper(s) to resolve current prompt continuity state + retained tail messages for a session
- `nanobot/agent/loop.py`
  - switch context rebuild path to fetch durable continuity state before prompt assembly

Concrete dependency unlocked by this mapping:
- task `.3` can now focus narrowly on the compaction execution path (foreground self-summary + reserve budget) without re-deciding prompt ordering semantics.

### 2026-03-12 implementation map — task .3 (agent-loop execution path / foreground self-summary)

Current execution path confirmed in code:
- normal automatic compaction is triggered from provider-reported usage in `nanobot/agent/loop.py`
- `_run_structured_compaction()` currently calls:
  - `compact_session(session=..., provider=self.background_provider, model=self.background_model, reserve_tokens=self._COMPACTION_RESERVE_TOKENS, ...)`
- `/new` archival summary generation also currently uses:
  - `generate_compaction_summary(snapshot, self.background_provider, self.background_model, ...)`
- `compact_session()` in `nanobot/session/compaction.py` currently performs both:
  - cut-point planning
  - summary generation
  - compaction entry append
narrowly through the passed provider/model pair

Primary architectural change for v1:
- normal compaction summary generation must move from **background provider/model** to the **foreground provider/model path that owns the session turn**
- memory extraction / MEMORY.md compaction may continue using the background path; that is a separate concern

Execution-path changes required:

1. **Split planning/persistence from summary-generation ownership**
   The current `compact_session()` helper is too bundled for the new architecture because it both chooses the cut point and runs summary generation with the supplied provider/model.

   Preferred refactor direction:
   - keep reusable cut-point / plan construction logic in `nanobot/session/compaction.py`
   - move actual summary-generation ownership decision into `AgentLoop._run_structured_compaction()`

   That likely means introducing a seam like:
   - `plan_compaction(...) -> CompactionPlan`
   - `apply_compaction_plan(... summary_text ...) -> persisted compaction event`

   rather than letting `compact_session()` remain the only orchestration unit.

2. **Foreground self-summary should run synchronously before the next normal turn continues**
   This already matches the current synchronous `_run_structured_compaction()` structure, which is good.

   Desired flow:
   - detect compaction trigger after provider response
   - compute compaction plan / cut point
   - generate structured summary using foreground provider + foreground model
   - persist compaction event + prompt summary state
   - run memory extraction separately
   - rebuild prompt from durable continuity state
   - continue normal conversation

3. **Reserve-budget enforcement must become explicit at the compaction invocation site**
   Right now `_COMPACTION_RESERVE_TOKENS` is passed through to `compact_session()` and used by `should_compact()`, but the v1 architecture needs a stronger contract:
   - reserve is not just a compaction threshold tuning knob
   - reserve is the guaranteed headroom required for the foreground model to summarize its own compacted span

   Therefore `_run_structured_compaction()` / its caller should explicitly ensure:
   - compaction is triggered early enough that foreground self-summary still fits
   - provider-usage-based compaction threshold remains derived from context window minus reserve
   - post-response compaction continues to be the normal path

4. **Keep provider-authoritative triggering; change only summary ownership**
   This thread does **not** change the earlier direction that normal compaction should trigger from provider-reported prompt usage rather than local pre-send estimates.

   So the execution target is:
   - trigger decision still comes from provider usage / overflow signals
   - but summary generation itself uses the foreground model, not the background model

5. **Degraded summary handling still preserves the same compaction flow**
   If foreground summary generation returns malformed/fallback-only output:
   - compaction may still complete if the continuity artifact contract is satisfied
   - retained literal tail remains available regardless
   - degraded summary status should be persisted in the compaction ledger / prompt state metadata

   This should not create a second compaction mode; it is the same flow with a degraded-quality recorded outcome.

6. **`/new` archival path can remain temporarily separate, but should be called out**
   `/new` currently archives the whole session using background summary generation.
   For this thread:
   - normal in-session compaction path is the priority target
   - `/new` archival can remain temporarily separate if needed
   - but it should be explicitly noted as follow-on cleanup so it does not silently retain old architecture forever

7. **Code surfaces likely to change**
   - `nanobot/agent/loop.py`
     - `_run_structured_compaction()` becomes the owner of foreground self-summary orchestration
     - automatic compaction trigger callsites stay provider-authoritative
     - `/new` may need a follow-on adjustment or TODO marker
   - `nanobot/session/compaction.py`
     - split bundled helper(s) so planning/cut-point logic is reusable without forcing background summary generation
     - return richer plan data needed by store persistence and prompt-state updates
   - `nanobot/session/store.py`
     - receive more explicit compaction-write inputs (summary status, tail bounds, reserve metadata)

8. **Sequencing note**
   The execution-path refactor in `.3` should land before the full regression suite mapping in `.4`, because the tests depend on whether compaction is still background-generated or truly foreground self-generated.

Concrete dependency unlocked by this mapping:
- task `.4` can now define tests against the intended single-path execution model instead of the old background-summary architecture.

### 2026-03-12 implementation map — task .4 (targeted regression tests)

Existing test anchors confirmed in repo:
- `tests/test_session_store.py`
- `tests/test_compaction_summary.py`
- `tests/test_compaction_integration.py`
- `tests/test_sqlite_integration.py`
- `tests/test_context_v2.py`
- `tests/test_context_prompt_cache.py`
- `tests/test_restart_resume.py`
- `tests/test_dashboard_api.py` (only if persistence shape changes reach inspection endpoints)

Preferred test strategy for this thread:
- extend the existing focused test files rather than creating a new mega-suite
- bias toward store + compaction + prompt-assembly seam tests
- add loop-level tests only where foreground-vs-background ownership must be proven

Targeted tests to add or update:

1. **SQLite schema / persistence tests** (`tests/test_session_store.py`)
   - `test_compaction_row_supports_summary_state_metadata_fields`
     - verifies additive `compaction` columns exist and can round-trip values for:
       - generation provider/model
       - reserve/input/output token metadata
       - continuity tail seq bounds
       - continuity/memory status fields
   - `test_prompt_summary_state_round_trips_and_rebuilds`
     - verifies `prompt_summary_state` can be written/read
     - verifies materialized state can be rebuilt from durable compaction rows when missing
   - `test_summary_rollup_event_is_recorded_on_working_to_distilled_rollover`
     - verifies rollover audit events persist when working summary exceeds budget and is consolidated

2. **Compaction summary generation / planning tests** (`tests/test_compaction_summary.py` and/or `tests/test_compaction_integration.py`)
   - `test_compaction_planning_returns_tail_bounds_and_previous_summary`
     - verifies the planning seam carries forward previous summary and retained tail range metadata
   - `test_degraded_summary_status_does_not_prevent_tail_persistence`
     - simulate malformed/fallback summary output
     - verify retained literal tail metadata still persists and compaction does not collapse continuity to summary text only
   - `test_foreground_self_summary_path_uses_foreground_provider_for_normal_compaction`
     - prove normal compaction summary generation uses foreground provider/model path
     - background provider remains available for memory extraction only

3. **Prompt assembly tests** (`tests/test_context_v2.py` or new focused context test file if needed)
   - `test_prompt_assembly_orders_distilled_then_working_then_literal_tail_then_history`
     - verify exact layer order in built messages
   - `test_prompt_assembly_uses_materialized_summary_state_when_present`
     - verify durable `prompt_summary_state` is preferred over metadata hacks
   - `test_prompt_assembly_rebuilds_summary_state_after_restart`
     - simulate missing in-memory state / fresh manager instance
     - verify rebuild from SQLite ledger + tail messages
   - `test_literal_tail_is_history_messages_not_system_summary_text`
     - verify retained tail stays as real user/assistant/tool message rows

4. **Loop / orchestration tests** (`tests/test_compaction_integration.py` or `tests/test_sqlite_integration.py`)
   - `test_provider_usage_trigger_runs_single_path_foreground_compaction`
     - when provider-reported usage crosses threshold, compaction runs once and stores foreground-generated summary metadata
   - `test_compaction_reserve_budget_is_applied_before_self_summary`
     - verifies reserve is treated as required headroom, not just incidental threshold math
   - `test_continuity_and_memory_outcomes_are_recorded_independently`
     - continuity succeeds / extraction fails
     - extraction succeeds / continuity degrades
     - neither outcome silently advances the other

5. **Optional inspection/API regression** (`tests/test_dashboard_api.py`) only if needed
   - `test_provider_call_detail_exposes_new_compaction_summary_layers`
   - only add this if the implementation changes dashboard/API payload shape for compaction detail inspection

Tests explicitly not required for first implementation pass:
- broad UI screenshot suite
- background fallback behavior
- generalized artifact graph behavior
- `/new` archival parity, unless touched by the implementation phase

Validation sequence once implementation begins:
1. most targeted store/schema tests
2. compaction summary/planning tests
3. prompt assembly tests
4. one or two loop/integration tests proving foreground ownership + independent outcomes
5. optional dashboard/API regression only if payload contract changes

Recommended proof points after code lands:
- one focused test proving foreground provider owns normal compaction summary generation
- one focused test proving degraded summary still leaves literal continuity tail available
- one focused test proving restart reconstruction from SQLite durable state

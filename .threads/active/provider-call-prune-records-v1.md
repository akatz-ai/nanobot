---
schema_version: 1
id: provider-call-prune-records-v1
title: "Provider Call / Tool Prune Records V1 — first-class prompt pressure, prompt assembly, and mutation audit trail"
status: active
priority: 1
created_at: '2026-03-12T04:34:00Z'
updated_at: '2026-03-12T06:06:00Z'
---

## Tasks
- [ ] provider-call-prune-records-v1.0 Lock repo-local acceptance and schema scope for first-class `provider_call`, `tool_prune_event`, and `prompt_assembly_snapshot` records, including how they relate to `message` and `compaction`
- [ ] provider-call-prune-records-v1.1 Define SQLite schema + persistence APIs for `provider_call` records (turn, iteration, model, finish_reason, provider token counts, produced message linkage)
- [ ] provider-call-prune-records-v1.2 Define SQLite schema + persistence APIs for `tool_prune_event` and `tool_prune_item` records (message/tool linkage, replacement mode, estimated savings, trigger context)
- [ ] provider-call-prune-records-v1.3 Define SQLite schema + persistence APIs for `prompt_assembly_snapshot` and call-scoped dynamic prompt artifacts needed for high-confidence replay
- [ ] provider-call-prune-records-v1.4 Wire loop/session write paths so provider responses, pruning mutations, and prompt assembly snapshots emit first-class records without regressing current sidecars
- [ ] provider-call-prune-records-v1.5 Expose read/query helpers for provider-call audit, prune/compaction correlation, prompt replay, and approximate turn-pressure analysis
- [ ] provider-call-prune-records-v1.6 Add targeted repo-local tests and dashboard/control-plane seam checks for auditability, replayability, and continuity-planning inputs

## Objective
Acceptance artifact: `acceptance/runtime/provider-call-prune-audit.feature`

Make prompt pressure, prompt assembly, and prompt-shaping mutations first-class persisted records so Nanobot can:
- audit what the provider actually saw on each call
- explain why prompt pressure rose or fell across turns
- correlate prompt growth with tool pruning and compaction
- support better continuity/retention planning for future compaction policy
- reconstruct, with high confidence, what the model saw for a provider call
- support future product claims around session/call auditability and incident review

## Design principles
1. **Provider calls are canonical usage truth**
   - usage, finish reason, and provider-observed prompt pressure belong to `provider_call`
2. **Prompt assembly must be replayable from normalized references**
   - do not rely on a giant duplicated prompt blob by default
3. **Pruning and compaction are first-class mutations**
   - they are not incidental implementation details
4. **Dynamic turn-scoped prompt artifacts must be pinned at call time**
   - retrieved memory and turn/session context cannot be inferred later from current state
5. **Hashes validate replay**
   - reconstruction should be provably consistent where possible
6. **Sidecars can remain during migration**
   - new DB records should not force immediate retirement of `.usage.jsonl`

## Proposed normalized record model

### Existing canonical records
- `session`
- `message`
- `compaction`

These remain foundational.

### New canonical records
- `provider_call`
- `tool_prune_event`
- `tool_prune_item`
- `prompt_assembly_snapshot`
- `retrieved_memory_snapshot`
- `turn_context_snapshot`

### Optional later records
- `prompt_artifact` (content-addressed dedup for reusable text artifacts)
- `provider_call_message` (join table if seq ranges are insufficient)
- `compaction_input_snapshot` (explicit replay of what was summarized during a compaction)

## Proposed schema candidates

### 1. `provider_call`
Canonical provider-observed prompt usage per LLM call.

Suggested fields:
- `id`
- `session_key`
- `call_index` (monotonic per session)
- `turn`
- `iteration`
- `provider_name`
- `model`
- `finish_reason`
- `input_tokens_raw`
- `cache_read_tokens`
- `cache_creation_tokens`
- `total_input_tokens`
- `output_tokens`
- `total_tokens`
- `context_window`
- `utilization_pct`
- `assembly_snapshot_id`
- `produced_message_seq_start`
- `produced_message_seq_end`
- `created_at`

Semantics:
- one row per provider response/call
- canonical source of operator/audit truth for prompt pressure
- linked to the normalized prompt assembly state used for the call
- linked to the messages produced by that call

### 2. `tool_prune_event`
First-class record for each pruning mutation pass that changes persisted prompt content.

Suggested fields:
- `id`
- `session_key`
- `turn`
- `iteration`
- `trigger_call_index`
- `reason`
- `strategy_version`
- `estimated_tokens_before`
- `estimated_tokens_after`
- `estimated_tokens_saved`
- `messages_pruned`
- `tool_results_pruned`
- `created_at`

Semantics:
- one row per pruning mutation pass
- records why pruning ran and how much it estimates it saved
- explains why later provider-observed prompt pressure may be lower than expected

### 3. `tool_prune_item`
Per-pruned result/message detail for audit and correlation.

Suggested fields:
- `id`
- `prune_event_id`
- `session_key`
- `message_seq`
- `tool_call_id`
- `tool_name`
- `replacement_kind`
- `original_content_chars`
- `estimated_tokens_removed`
- `created_at`

Semantics:
- one row per pruned tool result/message
- used to explain exactly which persisted content changed and by how much

### 4. `prompt_assembly_snapshot`
Normalized replay anchor for a specific provider call.

Suggested fields:
- `id`
- `session_key`
- `turn`
- `iteration`
- `provider_name`
- `model`
- `system_prompt_version`
- `system_prompt_hash`
- `memory_hash`
- `memory_chars`
- `compaction_id` (nullable)
- `compaction_summary_hash` (nullable)
- `message_seq_start`
- `message_seq_end`
- `prune_watermark_event_id` (nullable)
- `retrieved_memory_snapshot_id` (nullable)
- `turn_context_snapshot_id` (nullable)
- `assembled_prompt_hash`
- `assembled_prompt_tokens_est`
- `stable_prefix_tokens_est`
- `dynamic_turn_tokens_est`
- `visible_conversation_tokens_est`
- `created_at`

Semantics:
- does not duplicate the full prompt by default
- records which normalized artifacts and message range define the call payload
- allows high-confidence replay and hash validation

### 5. `retrieved_memory_snapshot`
Call-scoped dynamic memory injection artifact.

Suggested fields:
- `id`
- `session_key`
- `turn`
- `query_message_seq`
- `content_text`
- `content_hash`
- `item_count`
- `created_at`

Semantics:
- pins the exact retrieved-memory payload used for that call
- prevents replay drift from later retrieval changes

### 6. `turn_context_snapshot`
Call-scoped rendered session/channel/timestamp prepend artifact.

Suggested fields:
- `id`
- `session_key`
- `turn`
- `channel`
- `chat_id`
- `rendered_text`
- `content_hash`
- `created_at`

Semantics:
- pins transient turn-scoped context that would otherwise be unrecoverable later

## Relationships to existing records
- `message` remains the canonical conversation timeline.
- `compaction` remains the canonical major structural compression record.
- `provider_call` becomes the canonical provider-observed usage ledger.
- `tool_prune_event` / `tool_prune_item` become the canonical pruning audit trail.
- `prompt_assembly_snapshot` becomes the canonical normalized replay anchor for a provider call.

Expected correlations:
- `provider_call` explains observed prompt pressure at each call.
- `tool_prune_event` explains local prompt shrink operations before a call.
- `compaction` explains large structural compression events.
- `prompt_assembly_snapshot` explains exactly which session artifacts and message range were sent.
- Together they support:
  - prompt replay / “what the LLM saw”
  - approximate per-turn prompt pressure analysis
  - oversized-turn detection
  - future continuity-planning heuristics
  - audit/forensics for incident review

## Replay model

### Exact replay target
A provider call should be reconstructable from normalized records by:
1. loading `provider_call`
2. loading linked `prompt_assembly_snapshot`
3. resolving the referenced dynamic prompt artifacts
   - retrieved memory snapshot
   - turn context snapshot
4. resolving the referenced structural prompt artifacts
   - system prompt version/hash
   - memory snapshot/hash
   - compaction entry / summary hash
5. loading the referenced message sequence range
6. applying prune state up to the referenced prune watermark
7. assembling the provider payload in canonical order
8. validating the result against `assembled_prompt_hash`

### What exact replay requires
Exact replay is only possible when call-scoped prompt artifacts are pinned at persistence time.
A `provider_call` row alone is not enough.

## Acceptance proposal

### Repo-local acceptance to add

#### 1. Provider calls are persisted as first-class records
- each provider response creates a queryable `provider_call` record
- the record includes turn, iteration, model, finish reason, and provider-reported token counts
- the record links to the prompt assembly snapshot used for the call
- the record can be correlated to the messages produced by that call

#### 2. Tool pruning is persisted as first-class mutation history
- each pruning action creates a `tool_prune_event`
- pruned tool results/messages are queryable via per-item records
- estimated savings and replacement mode are visible to operators/debuggers

#### 3. Prompt assembly state is replayable from normalized call-scoped records
- each provider call has a corresponding prompt assembly snapshot or equivalent replay anchor
- dynamic turn-scoped artifacts are pinned so later system state changes do not invalidate replay
- replay can be hash-validated against the stored assembly hash

#### 4. Prompt pressure can be explained across turns
- a later reduction in provider-observed prompt pressure can be correlated to pruning or compaction records
- operators are not forced to infer prompt mutations from message content alone

#### 5. Continuity planning inputs are queryable
- the system can approximate the provider-observed prompt pressure contribution of recent turns
- future compaction retention planning can combine provider-call history with prune/compaction events
- the system does not claim exact intrinsic per-message token attribution from provider totals alone

## Phased implementation plan

### Phase 1 — provider truth + pruning audit
Goal: establish first-class provider and pruning records without changing replay/UI semantics broadly.

Deliver:
- `provider_call`
- `tool_prune_event`
- `tool_prune_item`
- write-path persistence from loop/session mutations
- targeted tests proving auditability and provider/prune correlation
- keep `.usage.jsonl` for compatibility and operational parity

### Phase 2 — normalized prompt replay anchor
Goal: make high-confidence replay possible for provider calls.

Deliver:
- `prompt_assembly_snapshot`
- `retrieved_memory_snapshot`
- `turn_context_snapshot`
- loop write-path linkage from provider call → assembly snapshot
- replay/query helpers for “what the LLM saw”
- targeted tests proving deterministic reconstruction + hash validation

### Phase 3 — optional dedup + deeper compaction audit
Goal: reduce duplication and complete the session audit graph.

Deliver (only if needed):
- `prompt_artifact` content-addressed dedup
- `provider_call_message` if seq ranges prove insufficient
- `compaction_input_snapshot` for exact compaction-input replay
- richer dashboard/control-plane audit APIs

## Migration notes
- Keep current sidecars (`.usage.jsonl`) during migration; do not force immediate retirement.
- Backfill is optional for older sessions; exact replay guarantees can start at migration cutover.
- For old sessions lacking call-scoped prompt artifacts, UI should clearly distinguish:
  - exact replay available
  - approximate reconstruction only
- Prefer additive migrations and write-both/read-new transitions where practical.

## Non-goals / not yet
- full dashboard UI design for these records
- final compaction continuity algorithm based on these records
- storing the full rendered provider payload blob for every call by default
- replacing all existing context/usage sidecars immediately
- broad cross-repo/shared-artifact widening before repo-local seam is proven

## Recommended next concrete step
Lock the repo-local schema + acceptance scope for Phase 1 and Phase 2, then do a bounded implementation pass for:
1. `provider_call`
2. `tool_prune_event`
3. `tool_prune_item`
4. write-path persistence + targeted tests

After that, advance to the replay seam with `prompt_assembly_snapshot` and call-scoped prompt artifacts.

## Notes
- This is repo-local planning work for now, but it advances the same Slice 1 session/context truthfulness seam.
- The likely architectural destination is a session record model where `message`, `provider_call`, `tool_prune_event`, `prompt_assembly_snapshot`, and `compaction` are the key auditable record types.
- Sidecars (`.usage.jsonl`) can remain during migration for backward compatibility and observability parity.
- The north-star product claim enabled by this design is: high-confidence auditability of what the model saw, what changed the prompt, and why prompt pressure changed across turns.

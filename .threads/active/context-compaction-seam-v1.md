---
schema_version: 1
id: context-compaction-seam-v1
title: "Context/Compaction seam \u2014 expose PromptAssemblyResult to API/dashboard"
status: active
priority: 1
created_at: '2026-03-11T21:28:24Z'
updated_at: '2026-03-11T23:30:37Z'
---

## Tasks
- [x] context-compaction-seam-v1.0 Lock shared seam scope: confirm which shared artifacts advance now (acceptance, contract, evals, fixtures) and record exact payload/visibility targets {claim_by=codex@sqlite-test claim_at=2026-03-11T21:32:14Z}
- [>] context-compaction-seam-v1.1 Implement nanobot backend/API exposure for PromptAssemblyResult-aligned compaction/context data, including pre/post snapshot provenance and manual /compact visibility {deps=[context-compaction-seam-v1.0] claim_by=codex@sqlite-test claim_at=2026-03-11T21:42:21Z}
- [x] context-compaction-seam-v1.2 Add/update repo-local API/serialization coverage for assembled prompt estimate, threshold, reserved headroom, stable vs dynamic sections, visible slice labeling, and pre/post snapshot provenance {deps=[context-compaction-seam-v1.1] claim_by=codex@nanobot claim_at=2026-03-11T23:25:57Z}
- [ ] context-compaction-seam-v1.3 Port the operator-facing context/compaction UI to `agentshq-app` using the canonical control-plane seam, keeping the legacy nanobot HTML dashboard deprecated/reference-only {deps=[context-compaction-seam-v1.2]}
- [ ] context-compaction-seam-v1.4 Run focused browser/manual validation in `agentshq-app` (Playwright/agent-browser as applicable), capture proof, and reconcile any shared artifact deltas before calling the slice complete {deps=[context-compaction-seam-v1.3]}

## Notes
### Task 1 progress

- Extracted the canonical backend/session context-inspection builder into `nanobot/session/context_inspection.py` so `dashboard/app.py` now delegates instead of reconstructing `promptAssembly` inline.
- `GET /api/context/{sessionId}` and the agent-scoped context routes now share the same builder path via `_build_live_context_response`.
- Tightened seam behavior to keep `triggerSnapshot` anchored to the pre-trigger snapshot while preferring the compaction log post snapshot for `postCompactionSnapshot`.
- Added focused builder/route regressions for required `promptAssembly` fields, stable vs turn-scoped section mapping, nullable provider-observed tokens, pre/post snapshot provenance, and session-scoped route behavior.
- Shared artifacts were re-checked for alignment, but not changed in this turn; this was repo-local preparatory work in service of the shared Slice 1 contract target.
- The legacy `nanobot/dashboard/static/index.html` surface is now considered deprecated/reference-only. Future operator-facing browser UI work for this seam belongs in `agentshq-app`, while nanobot remains the control-plane API and Discord runtime UI host.

### Task 0 seam lock

Minimal seam delta decision:
- Shared-artifact advancing slice, but surgical.
- The shared API contract already contains a skeletal `ContextInspectionResponse` / `PromptAssemblyResult` shape.
- Therefore the first-class shared artifact changes now are:
  1. `agentshq-platform/contracts/api/openapi.yaml` — tighten the existing schema to the minimal fields the nanobot API will actually expose.
  2. `agentshq-platform/fixtures/sessions/compaction/*` — add or update fixture coverage for below-threshold, triggered compaction, pre/post snapshots, and manual-compaction provenance.
- Shared acceptance features remain unchanged for now unless implementation shows insufficiency or contradiction.
- A premature manual-compaction acceptance scenario was removed during audit cleanup because it widened scope beyond the minimal seam task without explicit approval.
- Shared evals remain unchanged for now unless the exposed seam leaves ambiguity around provenance, snapshot source, or extraction-linkage wording.

Fields that must be exposed in the minimal seam:
- `sessionId`
- `promptAssembly.assembledPromptTokens`
- `promptAssembly.providerObservedPromptTokens` (nullable)
- `promptAssembly.compactionThresholdTokens`
- `promptAssembly.reservedHeadroomTokens`
- `promptAssembly.stablePrefixTokens`
- `promptAssembly.dynamicTurnTokens`
- `promptAssembly.visibleConversationSliceTokens`
- `promptAssembly.compactionTriggered`
- `promptAssembly.triggerSnapshot`
- `promptAssembly.preCompactionSnapshot`
- `promptAssembly.postCompactionSnapshot`
- `promptAssembly.sections` with enough section metadata to distinguish stable vs turn-scoped slices

Fields that should stay repo-local / deferred for later unless already trivial:
- broader extraction run linkage beyond current compaction/extraction logs
- new acceptance wording for pre-compaction orchestration details
- broad UI promises not already present in Slice 1 artifacts

Current artifact state vs seam:
- Acceptance artifacts are sufficient as behavioral targets.
- Contract exists but is not yet proven to match live nanobot API payloads; this is the artifact that must change first if names/shape differ.
- Evals can remain unchanged unless contract/payload review reveals missing operator-truth checks.
## OS Context
- Classification: Type B | Type C | Type D
- Relevant Slice: Slice 1
- Ownership Label: shared-artifact advancing
- Source Specs / Docs:
  - /data/projects/nanobot/docs/context-architecture-v2-spec.md
  - /data/projects/nanobot/docs/session-architecture-migration-spec-v2.md
  - /data/projects/nanobot/docs/cache-preserving-compaction-spec.md
- Shared Artifacts Checked:
  - /data/projects/agentshq-platform/acceptance/cross-product/explain-compaction.feature
  - /data/projects/agentshq-platform/acceptance/cross-product/inspect-context-from-dashboard.feature
  - /data/projects/agentshq-platform/acceptance/cross-product/compaction-summary-boundary.feature
  - /data/projects/agentshq-platform/contracts/api/openapi.yaml
  - /data/projects/agentshq-platform/evals/cross-product/context-accuracy.yaml
  - /data/projects/agentshq-platform/evals/cross-product/compaction-summary-integrity.yaml
- Shared Artifacts To Update:
  - agentshq-platform/contracts/api/openapi.yaml
  - agentshq-platform/fixtures/sessions/compaction/*
  - agentshq-platform/evals/cross-product/context-accuracy.yaml (only if seam payload/provenance needs stronger evaluation wording)
  - agentshq-platform/evals/cross-product/compaction-summary-integrity.yaml (only if seam payload/extraction-status visibility needs stronger evaluation wording)
  - acceptance feature text only if seam scope reveals an insufficiency or contradiction
- manual-compaction continuity/provenance acceptance remains deferred to a follow-on task unless UI/runtime work in this slice makes it unavoidable
- Repos Touched:
  - /data/projects/nanobot
  - /data/projects/agentshq-platform
- Validation Required:
  - nanobot backend/context/compaction control-plane API tests
  - contract-aware API validation against the exposed prompt-assembly seam
  - `agentshq-app` UI/workflow validation for the operator-facing browser surface
  - focused manual proof of compaction explanation + time-travel context view
- Human Approval Needed:
  - Yes, before changing acceptance wording or materially widening the shared API contract beyond the minimal seam delta
- Exit Criteria:
  - PromptAssemblyResult-aligned data is exposed through the nanobot control-plane API seam
  - operator-visible compaction/context fields in `agentshq-app` satisfy Slice 1 shared artifacts
  - pre/post compaction snapshots are distinguishable in operator-visible surfaces
  - manual /compact and automatic compaction share the same visible provenance path
- contract/fixture changes are minimal and explicit

### 2026-03-11 backend seam extraction

- This increment stays Slice 1 and keeps acceptance scope unchanged.
- Ownership for this code change is repo-local preparatory: nanobot now has a backend/session-level canonical context inspection builder while still targeting the shared `GET /api/context/{sessionId}` contract.
- Implemented a new backend seam in `nanobot/session/context_inspection.py` that assembles the contract-aligned `ContextInspectionResponse` / `promptAssembly` payload from session bundle, context log, and compaction log artifacts.
- `nanobot/dashboard/app.py` now delegates session and agent-scoped context inspection responses to that backend builder instead of reconstructing prompt-assembly payloads inline.
- Added focused regression coverage for required promptAssembly fields, section stable/turn-scoped mapping, pre/post snapshot precedence, nullable provider-observed tokens, and session-scoped route delegation.
- Shared artifacts were re-checked before the extraction and remain unchanged in this increment:
  - `/data/projects/agentshq-platform/contracts/api/openapi.yaml`
  - `/data/projects/agentshq-platform/docs/playbooks/first-vertical-slice-map.md`
  - `/data/projects/agentshq-platform/fixtures/sessions/compaction/README.md`
- Deliberately deferred:
  - dashboard UI consumption changes beyond existing route compatibility
  - acceptance wording changes
  - broader refactors outside the canonical context-inspection seam
  - repo-local tests and focused UI/manual validations pass

### 2026-03-11 provider-authoritative compaction/status follow-up

- New repo-local acceptance direction has now been set in `/data/projects/nanobot/acceptance/discord/system-status-context-threshold.feature`.
- This follow-up remains Type B / Slice 1 / shared-artifact advancing, but the immediate implementation step is repo-local within `nanobot`.
- Product contract clarified:
  - Discord system status must show the latest persisted **provider-reported total input tokens**.
  - Normal compaction must trigger **immediately after** a provider response whose total input tokens exceed the threshold.
  - Normal compaction must **not** trigger from local pre-send estimates.
  - Provider overflow error is the only pre-response recovery signal and should trigger compaction + one retry.
  - Internal token estimation should no longer act as a first-class trigger path for compaction or operator-visible status.
- Intended implementation seam:
  - `nanobot/agent/loop.py` — remove normal estimate-driven pre-send compaction from `_finalize_prompt_for_send(...)`; keep provider-overflow recovery; trigger compaction from post-response provider usage.
  - `nanobot/discord/system_status.py` — tighten canonical displayed token source to provider-backed usage.
  - `nanobot/session/compaction.py` — narrow normal compaction decision inputs to provider-backed usage / overflow recovery and remove estimate-driven trigger behavior where practical.
- Required validation for the follow-up implementation:
  - `pytest -q tests/test_system_status_dashboard.py`
  - targeted compaction tests in `tests/test_compaction_integration.py`
  - confirm no pre-response estimate-only compaction remains in the normal path
  - confirm provider overflow still compacts and retries once

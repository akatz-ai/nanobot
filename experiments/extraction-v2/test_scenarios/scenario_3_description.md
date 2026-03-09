# Scenario 3: Supersede/Update (Outdated Memories)

## Description

A conversation about infrastructure updates for the comfygit project. The memory graph has stale facts; the conversation reveals multiple replacements. This tests the `supersede` action path.

## What It Tests

- Whether Haiku correctly identifies outdated facts and outputs `supersede` actions
- Whether the supersede items correctly identify WHAT they're replacing
- Whether new facts (no existing counterpart) get plain `add` actions
- Mixed scenario: some supersedes + some adds

## Existing Memories (5 stale items)

1. "comfygit deployed on Heroku" → SUPERSEDED by Fly.io
2. "comfygit API is on v2" → SUPERSEDED by v3
3. "comfygit uses Redis for job queue" → SUPERSEDED by pgqueue
4. "comfygit database is AWS RDS" → SUPERSEDED by Neon serverless Postgres
5. "workflow endpoint is /workflows" → SUPERSEDED by /environments

## Expected Extractions

- **supersede** — Deployment moved from Heroku to Fly.io (iad region primary)
- **supersede** — API moved from v2 to v3 (v2 returns 410 Gone)
- **supersede** — Job queue moved from Redis to pgqueue (Python library)
- **supersede** — Database moved from RDS to Neon serverless Postgres
- **supersede** — Workflow endpoint renamed /workflows → /environments
- **add** (optional) — v3 migration guide at docs.comfygit.org/migration/v3
- **add** (optional) — v3 auth changed to OAuth2 PKCE

## Pass Criteria

- At least 4 supersede actions for the infrastructure migrations
- Each supersede includes `supersedes_content` matching the outdated memory
- No items that just re-add the same stale facts
- New info (migration guide URL, OAuth2 PKCE) captured as `add` if included

## Failure Modes to Watch For

- Using `add` instead of `supersede` for updated facts (ignoring existing memories)
- Superseding a fact but getting the `supersedes_content` wrong
- Missing infrastructure migrations entirely
- Superseding the Singapore region note (that was never decided, shouldn't be extracted)

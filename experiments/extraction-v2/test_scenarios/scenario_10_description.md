# Scenario 10: Agent-Scoped vs Shared Memories

## Description

A conversation that explicitly mixes two categories: project-wide facts (shared, all agents need to know) and agent-specific behavior preferences (scoped to this specific agent). Tests whether Haiku correctly assigns `scope` values.

## What It Tests

- Correct `scope` assignment: `shared` for project facts, `agent` for agent-specific preferences
- Whether Haiku distinguishes between "facts about the project" and "instructions for how I should behave"
- Precision on scope: agent preferences must not be marked `shared`

## Expected Extractions with Correct Scopes

**Scope = "shared" (project-wide, any agent should know):**
1. **fact** — GitHub org: github.com/acme-corp (all repos)
2. **fact** — Primary language: Python; TypeScript for frontend only
3. **fact** — Environments: staging.acme-corp.com and acme-corp.com (prod), separate DBs
4. **fact/procedure** — On-call: 2-week rotation, alerts in #platform-alerts

**Scope = "agent" (this specific agent's behavior):**
5. **preference** — Always run tests after code changes without being asked
6. **preference** — Commit messages: conventional commits format (type(scope): message)
7. **preference** — Response style: short, direct, bullets only for 3+ items, no preamble
8. **preference** — Code review order: security → correctness → style

## Pass Criteria

- Items 1-4 have `scope: "shared"`
- Items 5-8 have `scope: "agent"`
- No cross-contamination (agent preferences marked as shared, or project facts marked as agent)
- All 8 items extracted (full recall)

## Failure Modes to Watch For

- Marking all items as `scope: "shared"` (not distinguishing agent preferences)
- Marking all items as `scope: "agent"` (not recognizing project-wide facts)
- Mixing response style preferences with project facts
- Missing the code review priority order
- Missing the on-call rotation fact

# Scenario 7: Multi-Topic Conversation

## Description

A conversation that explicitly jumps between 4 distinct, unrelated topics: CI/CD setup, frontend stack, API rate limiting, and feature flag naming. Each topic has 1-2 extractable facts.

## What It Tests

- Whether Haiku handles context switching cleanly across topics
- Whether it captures all topics without confusing them
- Whether items from different domains are correctly categorized
- No cross-contamination between topics

## Expected Extractions (7-9 items)

**CI/CD topic:**
1. **procedure/decision** — GitHub Actions pipeline: lint → test → build docker → push GHCR → deploy Fly.io (main only)
2. **decision** — Python matrix testing: versions 3.11 and 3.12

**Frontend topic:**
3. **decision** — Frontend uses Next.js 14 App Router, TypeScript only (no plain JS)
4. **decision** — UI design system: shadcn/ui + Tailwind CSS (no MUI/Chakra)

**Rate limiting topic:**
5. **decision** — Per-user rate limits: 100 req/min (free tier), 1000 req/min (paid tier)
6. **decision** — Rate limiting via Redis sliding window algorithm

**Feature flags topic:**
7. **procedure** — Feature flag naming convention: {team}_{feature_name}_{state} (e.g., payments_new_checkout_enabled)
8. **fact** — Feature flags managed via LaunchDarkly

## Pass Criteria

- At least 6 items extracted
- Items from all 4 topic areas represented
- No confusion between topics (e.g., applying CI/CD details to frontend)
- Rate limit numbers (100/1000) correctly captured
- Naming convention format correctly captured

## Failure Modes to Watch For

- Only extracting 1-2 items total (under-extraction from topic switching)
- Conflating rate limits with feature flags
- Missing the Python version matrix requirement
- Not capturing the "TypeScript only, no plain JS" constraint

# Scenario 9: Dense Decision-Making

## Description

A rapid-fire architectural decision session with 12 distinct decisions made in 24 messages. Every exchange produces a durable decision. This tests whether Haiku can handle high-density extraction without dropping items or hallucinating.

## What It Tests

- Whether Haiku can extract ALL decisions from a dense session
- Whether importance scores are appropriately varied (some are "non-negotiable", others are less critical)
- Whether each decision is captured as a distinct item, not combined
- Recall: does it miss any of the 12 decisions?

## Expected Extractions (10-12 items, all `add`)

1. **decision** — Architecture: modular monolith, extract services only when there's a clear scaling need
2. **decision** — API: GraphQL with Apollo Server (multi-client: web, mobile, third-party)
3. **decision** — Auth: OAuth2 PKCE, no API key auth for first-party clients, JWT RS256
4. **decision** — Search: Elasticsearch (TypeSense too limited for complex FTS requirements)
5. **decision** — Caching: Redis with 1h default TTL, 24h for user profiles
6. **decision** — Background jobs: Celery + Redis broker + Celery Beat (no separate job service)
7. **decision** — File storage: Cloudflare R2 (S3-compatible) + Cloudflare CDN, no AWS S3 (egress costs)
8. **decision** — Observability: structured JSON logs → Grafana Loki, traces via OpenTelemetry → Grafana Tempo
9. **decision** — IaC: Terraform for all cloud resources, no manual console changes
10. **decision** — Secrets: HashiCorp Vault (non-negotiable), no env var secrets
11. **fact** — Error tracking: Sentry (already paid for), integrate all services
12. **procedure** — PR policy: 2 approvals required, no self-merge on main, feature flags for risky deploys

## Pass Criteria

- At least 10 items extracted
- Each of the 12 decision areas represented
- Non-negotiable items (Vault secrets policy) get higher importance score (0.9+)
- No fabricated decisions not in the conversation

## Failure Modes to Watch For

- Combining multiple decisions into one vague item
- Dropping decisions (especially later ones as context gets long)
- Missing the "non-negotiable" emphasis on Vault
- Over-extracting assistant background knowledge as decisions

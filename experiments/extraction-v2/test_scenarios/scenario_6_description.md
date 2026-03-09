# Scenario 6: Contradictory Information (Mind Changes)

## Description

A conversation where decisions are made and then reversed mid-conversation. The user initially picks PostgreSQL then switches to ClickHouse; initially picks Kafka then switches to Redpanda. The final decisions are what matter.

## What It Tests

- Whether Haiku extracts the FINAL decision, not the reversed one
- Whether it handles the "actually, let's do X instead" pattern correctly
- Whether intermediate rejected options are ignored
- Final state extraction: what was actually decided

## Expected Extractions (3-4 items)

1. **decision** — Analytics service uses ClickHouse (NOT PostgreSQL — that was reversed)
2. **decision** — Event streaming uses Redpanda (Kafka-compatible, NOT Apache Kafka — reversed)
3. **decision** — Analytics dashboard uses Grafana (already running for infra metrics)
4. **fact** (optional) — Full analytics stack: Redpanda → ClickHouse → Grafana

## NOT Expected to Extract

- "PostgreSQL chosen for analytics service" (reversed)
- "Apache Kafka for event streaming" (reversed)
- Both the initial and revised decisions as separate items

## Pass Criteria

- Exactly 0 items mentioning PostgreSQL as the chosen analytics database
- Exactly 0 items mentioning Apache Kafka as the streaming solution
- ClickHouse and Redpanda correctly captured as the final decisions
- No "both were considered" hedging in the extracted content

## Failure Modes to Watch For

- Extracting both PostgreSQL AND ClickHouse as decisions (missing the reversal)
- Extracting both Kafka AND Redpanda
- Extracting "PostgreSQL ruled out" as a separate item (redundant noise)
- Extracting the evaluation rationale for each option rather than the decision

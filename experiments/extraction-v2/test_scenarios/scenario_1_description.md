# Scenario 1: Clean Extraction (No Existing Memories)

## Description

A simple, focused technical discussion where a user and assistant plan a new Python web project from scratch. The conversation is clean, structured, and contains clear preferences and decisions.

## What It Tests

- Basic extraction quality with no existing memories to compare against
- Whether Haiku correctly identifies preferences, decisions, facts, and goals
- Whether all distinct items are captured without hallucination
- Correct scope assignment (most should be `shared`, project-level facts)

## Expected Extractions (5-8 items, all `add` action)

1. **decision** — Project uses FastAPI as the web framework
2. **decision** — PostgreSQL for database, SQLAlchemy for ORM, Alembic for migrations
3. **preference** — User prefers Pydantic v2 for data validation (performance over migration pain)
4. **preference** — User prefers uv for package management over pip/poetry
5. **decision** — Project structure: app/models, app/schemas, app/routers, app/services pattern
6. **preference** — pytest for testing with minimum 80% coverage requirement
7. **decision** — Deploy with Docker to Fly.io
8. **decision** — GitHub Actions CI pipeline with ruff lint + pytest + Docker build
9. **fact** — Project name is "taskflow-api"

## Pass Criteria

- At least 6 items extracted
- No items that are pure conversational filler (greetings, acknowledgments)
- All items have `action: "add"`
- Preferences are correctly typed as `preference`, not `fact`
- Coverage requirement (80%) is captured with the preference
- No hallucinated facts not present in the conversation

## Failure Modes to Watch For

- Extracting "FastAPI has automatic OpenAPI docs" as a fact (this is assistant background knowledge, not a user decision)
- Missing the 80% coverage requirement
- Combining multiple distinct decisions into one vague item

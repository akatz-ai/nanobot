# Scenario 12: Taste Signal Extraction — Architecture & Code Review Corrections

## Description

A code review conversation where the user corrects multiple aspects of a PR: unnecessary dependencies, error handling patterns, API response structure, and test coverage philosophy. Tests whether the extractor distinguishes between project-specific decisions and generalizable taste principles.

## What It Tests

- Whether architecture/code-style corrections are extracted as taste signals
- Whether the extractor generalizes from specific corrections ("don't use python-pushover") to principles ("check existing utils before adding dependencies")
- Whether API design taste is captured as a reusable principle
- Whether testing philosophy is captured as taste
- Mix of taste signals and regular decisions in the same conversation

## Expected Extractions (5-9 items)

### Taste signals (type: "taste"):
1. **taste [architecture]** — Always check existing utilities/services before adding new dependencies
2. **taste [code-style]** — Use specific exception types, not broad Exception catches. Errors should be visible to users, not silently logged
3. **taste [api]** — All API responses must be JSON objects with 'status' as the first field. Never return bare strings or unstructured responses
4. **taste [code-style]** — Every error response code must have a corresponding test. No untested error paths
5. **taste [code-style]** — Tests should assert on full response body structure (JSON shape, status field, error codes), not just HTTP status codes

### Regular extractions:
6. **decision** — Notification service uses configurable retry: max_retries=3, base_delay=1.0, backoff_factor=2.0
7. **decision** — Shared response helper created in utils/responses.py with success_response() and error_response()

## Pass Criteria

- At least 3 items have type "taste"
- The dependency principle is generalized (not just "don't use python-pushover" but "check existing utils first")
- API response structure is captured as a hard rule, not a suggestion
- Testing philosophy captures both "test every error path" AND "assert on response structure"
- Regular decisions (retry config, response helper location) are NOT typed as "taste"
- Taste items include domain tags ([architecture], [api], [code-style])

## Failure Modes to Watch For

- Extracting "removed python-pushover dependency" as a fact instead of generalizing to the principle
- Missing the testing philosophy (it's spread across two messages)
- Typing the API response rule as "preference" instead of "taste"
- Not distinguishing between the generalizable principle and the specific implementation decision
- Extracting the response helper location (utils/responses.py) as taste instead of decision

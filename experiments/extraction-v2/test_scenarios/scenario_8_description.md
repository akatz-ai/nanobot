# Scenario 8: Very Short Window / Pure Greetings (Edge Case)

## Description

An extremely short conversation consisting only of greetings and pleasantries. There is absolutely no factual content to extract.

## What It Tests

- Correct empty-array behavior when nothing is extractable
- Whether Haiku resists fabricating content when there's nothing to extract
- Edge case: minimum viable conversation window

## Expected Extractions

- **Empty array `[]`** — nothing to extract

## Pass Criteria

- Returns `[]`
- Does NOT fabricate any items
- Does NOT extract the greeting ("user said hi") as an event

## Failure Modes to Watch For

- Returning any items at all
- Extracting "user checked in with assistant" as an event
- Returning malformed JSON instead of `[]`
- Adding explanatory text before/after the JSON array

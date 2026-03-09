# Scenario 2: Heavy Dedup (Many Existing Memories Overlap)

## Description

A check-in conversation about the nanobot memory system. The graph already knows most of the facts being discussed. Only one item is genuinely new: the upgrade from Haiku 3.5 to Haiku 4.5.

## What It Tests

- Whether Haiku respects existing memories and skips duplicates
- Whether genuinely new information is still extracted
- Whether Haiku correctly identifies a `supersede` situation (model upgrade)
- Precision: does it avoid re-extracting things already in the graph?

## Existing Memories (6 items pre-loaded)

1. LanceDB + text-embedding-3-small 512 dims (exact match with conversation)
2. Dedup thresholds 0.95/0.85 (exact match)
3. Extraction trigger conditions (exact match)
4. Watermark guardrail (exact match)
5. Background model is claude-haiku-3-5 **(OUTDATED — conversation says upgrade to 4.5)**
6. XML-tagged transcript format (exact match)

## Expected Extractions

- **supersede** — Background extraction model updated from claude-haiku-3-5 to claude-haiku-4-5
- Items 1-4 and 6: should be **skipped** (already in graph, identical content)
- Possibly 0-1 additional items if there's something genuinely new

## Pass Criteria

- Exactly 1 item extracted (the model upgrade supersede)
- OR at most 2 items if Haiku reasonably considers the model rationale new info
- No items that duplicate existing memories verbatim
- The supersede item correctly identifies what it replaces

## Failure Modes to Watch For

- Re-extracting all facts from the conversation despite identical existing memories (precision failure)
- Missing the model upgrade (recall failure on the only genuinely new item)
- Generating a supersede that misidentifies what's being replaced

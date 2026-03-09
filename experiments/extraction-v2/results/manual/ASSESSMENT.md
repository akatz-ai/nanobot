# Manual Verification of V3 Extraction Prompt

**Date:** 2026-03-06
**Reviewer:** nanobot-dev agent (manual inspection of Haiku outputs)

## Test Matrix

| Scenario | Description | Items | Time | Status | Notes |
|----------|-------------|-------|------|--------|-------|
| S01 | Clean extraction | 10 | 5.1s | ✅ PASS | All adds, correct types |
| S02 | Heavy dedup | 1 | 1.3s | ✅ PASS | Only new fact extracted, 5 existing skipped |
| S03 | Supersede/update | 6 | 4.9s | ✅ PASS | 5 supersedes with correct targets + 1 add |
| S04 | Noisy/ambiguous | 3 | 2.8s | ✅ PASS | No noise (cat/coffee/cursor/tomorrow filtered) |
| S05 | Code-heavy | 4 | 2.4s | ⚠️ MINOR | Zero code, 1 borderline config value |
| S06 | Contradictions | 4 | 3.1s | ✅ PASS | Only final decisions (ClickHouse, Redpanda) |
| S07 | Multi-topic | 8 | 4.4s | ✅ PASS | All 4 topics covered |
| S08 | Empty/greeting | 0 | 0.5s | ✅ PASS | Perfect empty array |
| S09 | Dense decisions | 12 | 6.1s | ✅ PASS | 12/12 decisions captured |
| S10 | Scope routing | 8 | 4.4s | ✅ PASS | 4 shared + 4 agent, correct routing |
| Real | Live session data | 11 | 11.5s | ✅ PASS | No roleplaying, clean JSON |

## Key Observations

### Strengths
1. **Zero roleplaying failures** — The XML tags + "NOT a conversational agent" framing is bulletproof
2. **Excellent noise filtering** — The "3-months-from-now" heuristic works perfectly
3. **Accurate supersede targeting** — Haiku correctly identifies which existing memory to replace
4. **Perfect dedup** — When shown 6 existing memories, extracted only 1 genuinely new item
5. **Correct contradiction handling** — Only final decisions, never reversed ones
6. **Clean scope assignment** — Agent preferences vs shared facts correctly routed

### Minor Issues
1. **S05 Item 4**: `AUTH_PROVIDER='internal'` is a borderline implementation artifact. The embedding dedup layer would catch this if there's already a memory about auth provider choice.
2. **S01 density**: 10 items from one conversation could be argued as over-extraction, but each passes the 3-months test.

### Cost Analysis
- Average: 1,623 input + 576 output tokens per extraction
- Average cost: ~$0.004 per extraction run
- Average latency: 3.5s (synthetic), 11.5s (real 40-msg window)

## Production Recommendation

**V3 prompt is approved for production.** The extraction quality is consistently high across all tested scenarios. The minor issues (1 borderline item in 56 total extractions = 1.8% rate) are acceptable and will be caught by the post-extraction embedding dedup layer.

### Guardrails to implement in production code:
1. JSON parse error → don't advance watermark, retry next trigger
2. Max 25 items per extraction → log warning if exceeded
3. Strip markdown fences from output before parsing
4. Validate action ∈ {add, supersede}, scope ∈ {shared, agent}
5. Clamp importance to [0.0, 1.0]
6. Run embedding dedup as post-extraction filter before graph commit

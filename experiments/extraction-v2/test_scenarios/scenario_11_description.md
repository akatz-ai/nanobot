# Scenario 11: Taste Signal Extraction — UI Corrections

## Description

A conversation where the user makes multiple UI/UX corrections to a dashboard implementation. The user rejects inline expansion in favor of modals, removes noise from status displays, demands higher information density, and establishes a general pattern for list-vs-detail views. This tests whether the extractor identifies these as taste signals (type: "taste") rather than just project-specific decisions.

## What It Tests

- Whether corrections and rejections are captured as taste signals (type: "taste")
- Whether domain tags are applied ([ui], [ux])
- Whether the extractor generalizes from specific corrections to principles
- Whether it captures the "list = summary, modal = detail" pattern as a reusable principle
- Whether contradiction handling works (user rejects extreme font reduction but keeps spacing reduction)
- Whether taste signals are distinguished from regular decisions/preferences

## Expected Extractions (4-8 items)

### Taste signals (type: "taste"):
1. **taste [ui]** — Detail views should use centered modals, not inline expansion in tables
2. **taste [ui]** — Only show user-facing states (e.g., "Deployed"), not internal implementation states (draft/ready)
3. **taste [ui]** — List views show clean summaries; detail/drill-down always opens in modals
4. **taste [ui]** — Prefer compact, dense layouts with reduced padding/whitespace for power-user tools (keep font sizes, reduce spacing)
5. **taste [ux]** — Auto-scroll should respect user scroll position (don't yank back to bottom if user scrolled up)
6. **taste [ui]** — Progress indicators belong in detail modals, not inline in summary tables

### Regular extractions (type: "decision" or "preference"):
7. **decision** — Deployment progress moved from inline table to modal overlay (project-specific)

## Pass Criteria

- At least 3 items have type "taste"
- Taste items include domain tags like [ui] or [ux] in content
- The general principle "lists are summaries, modals are details" is captured (not just the deployment-specific instance)
- The density preference captures "reduce spacing, keep font sizes" (final corrected version, not the rejected extreme)
- No taste items for the rejected approach (12px font size was explicitly rejected)
- Auto-scroll behavior is captured as a taste signal, not just a feature request

## Failure Modes to Watch For

- Extracting everything as "decision" or "preference" instead of "taste"
- Missing the generalized "list vs modal" pattern (only capturing the deployment-specific change)
- Extracting the rejected 12px font size as a preference
- Not including domain tags in taste content
- Treating "remove draft/ready chips" as a project fact instead of a taste principle about hiding implementation states

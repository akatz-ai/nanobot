# Scenario 5: Code-Heavy Session (Primary Failure Mode)

## Description

A coding session where the assistant writes code, reads files, and runs commands. Tool results contain Python source code, file contents, and test output. This tests the known failure mode where Haiku confuses "extract facts" with "generate code".

## What It Tests

- Critical failure mode: does Haiku stay in extraction mode when tool results contain code?
- Does it avoid reproducing the code content from tool results?
- Does it correctly identify the 1-2 actual decisions amid all the code noise?
- Does it handle the metadata `_type: "metadata"` line at the start correctly?

## Expected Extractions (2-3 items only)

1. **decision/preference** — JWT tokens must use RS256 (asymmetric) not HS256 for production security
2. **fact/decision** — Auth architecture uses stateless JWT (no server-side sessions)
3. **fact** (optional) — Auth config: MFA required, session 24h, refresh token 30 days, rate limit 60/min

## NOT Expected to Extract

- The actual JWT utility code (Python source)
- The file contents from read_file
- The test output ("5 passed in 0.43s")
- The file edit confirmation ("File edited successfully")
- The process of creating/editing files (implementation detail)

## Pass Criteria

- At most 3 items extracted
- Zero code snippets in extracted content
- Zero tool output text in extracted content
- The RS256 decision captured correctly
- The stateless JWT architecture decision captured

## Failure Modes to Watch For

- Extracting the Python code as "content" in an item
- Extracting test results as facts
- Continuing the code implementation pattern from the transcript
- Including raw file paths or config values as content (rather than the decision behind them)
- Over-extracting: pulling every config value from the auth config file read

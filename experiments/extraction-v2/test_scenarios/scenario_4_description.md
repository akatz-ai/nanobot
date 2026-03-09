# Scenario 4: Ambiguous/Noisy Conversation

## Description

A debugging session with significant off-topic content: tangential discussion about a Cursor feature, casual conversation ("my cat knocked over coffee"), and extensive debugging details that are session-specific. Only 2-3 genuinely durable facts are buried in the noise.

## What It Tests

- Whether Haiku ignores transient chatter and debug noise
- Whether only genuinely extractable facts are captured
- Precision: does it avoid extracting session-specific debugging details?
- Whether it captures important decisions made amid the noise

## Expected Extractions (1-3 items only)

1. **decision/preference** — Token storage being migrated from AsyncStorage to expo-secure-store (encrypted)
2. **decision** — Promise-based mutex added for token refresh to prevent race condition on background resume
3. **fact** — App minimum versions: iOS 16, Android API 29 (set in stone)

## NOT Expected to Extract

- The debugging details about the 401 error (session-specific)
- The race condition explanation (background knowledge, not a project fact)
- The Cursor multi-file edit feature (tangential, off-topic)
- "My cat knocked over coffee" (obviously not extractable)
- "I'll do this tomorrow morning" (ephemeral intent, not a durable fact)

## Pass Criteria

- At most 3 items extracted
- No session-specific debugging chatter extracted
- No Cursor discussion extracted
- No ephemeral intent ("I'll do this tomorrow") extracted
- Platform minimum versions captured (this is explicitly "set in stone")

## Failure Modes to Watch For

- Over-extraction: pulling in the race condition explanation or debugging steps as "facts"
- Extracting "I'll do this tomorrow morning" as a goal/plan
- Extracting the Cursor feature discussion
- Missing the platform minimum versions (buried at the end)

Feature: Compaction summary ledger preserves continuity truthfully and auditably

  Background:
    Given structured compaction is enabled
    And compaction summary generation is performed by the foreground agent that owns the session
    And background-model fallback summarization is out of scope for this version

  Scenario: Foreground agent generates its own compaction summary when threshold is exceeded
    Given the latest provider-reported input usage exceeds the compaction threshold
    And the foreground model still has the configured compaction summary reserve available
    When compaction runs for that session
    Then the foreground agent generates the compaction summary itself
    And the summary is persisted as a first-class ledger artifact
    And the system does not require a separate background summarizer for normal compaction

  Scenario: Compaction requires reserved summary headroom
    Given compaction decisions are based on provider-reported prompt usage
    When the system evaluates whether normal compaction can run
    Then it accounts for the configured summary reserve required by the foreground model
    And it does not wait so long that the foreground model lacks space to summarize its own compacted span

  Scenario: Summary ledger is stored as queryable session data in SQLite
    Given a compaction run succeeds
    When the system persists compaction results
    Then the compaction run is recorded in SQLite as first-class session data
    And the resulting summary artifact is recorded in SQLite as first-class session data
    And the operator can correlate the summary artifact to the summarized session range and compaction run

  Scenario: Compaction event, summary artifact, and prompt-facing state remain independently inspectable
    Given a compaction run succeeds
    When the operator inspects durable continuity records for that session
    Then the compaction event metadata is inspectable independently of the summary text artifact
    And the summary text artifact is inspectable independently of the prompt-facing materialized summary state
    And the current prompt-facing working and distilled summary state can be traced back to durable compaction records

  Scenario: Prompt-facing summary state is separated into working and distilled layers
    Given multiple compaction summaries exist for a session
    When the system prepares prompt-facing continuity state for a future turn
    Then it may inject a recent working summary layer for near-term continuity
    And it may inject a distilled summary layer for long-range continuity
    And the full historical summary ledger is not injected blindly into the prompt

  Scenario: Working summary rollover is governed by token budget rather than blind overwrite
    Given the prompt-facing working summary grows beyond its configured token budget
    When the system updates prompt-facing summary state after a new compaction
    Then older working-summary content is consolidated into the distilled summary layer
    And newer compaction summary content remains available in the working summary layer
    And the system does not discard older summary history without preserving it in the ledger

  Scenario: Recent literal continuity is preserved alongside summary state
    Given compaction removes older transcript history from the active window
    When the next prompt is assembled after compaction
    Then the prompt includes bounded summary state for continuity
    And the prompt also retains a configured recent literal message tail
    And the system does not rely on summary text alone for immediate conversational continuity

  Scenario: Malformed summary output does not erase recent high-signal continuity
    Given compaction runs on a session with recent high-signal user turns in the retained continuity window
    And structured summary generation returns malformed, empty, or fallback-only output
    When the next prompt is assembled after compaction
    Then the prompt still retains a bounded literal continuity tail containing recent user task direction
    And generic fallback text is not the only continuity artifact available to the next turn
    And the operator can inspect that summary generation degraded without losing the retained literal continuity range

  Scenario: Continuity summary and memory extraction are tracked as separate outcomes
    Given a compaction span is selected
    When compaction and memory extraction run on that span
    Then continuity summary generation is tracked independently from memory extraction
    And failure in memory extraction does not silently erase or roll back the continuity summary outcome
    And failure in continuity summary generation does not falsely advance the memory extraction checkpoint

  Scenario: Compaction records expose an auditable summarized range and kept range
    Given a compaction run completes
    When the operator inspects compaction records for that session
    Then the records identify the summarized session range
    And the records identify the first kept message or entry after compaction
    And the records identify the literal continuity tail retained for the next prompt
    And the records identify the summary artifact used for post-compaction continuity

  Scenario: Prompt-facing summary state is reconstructed from durable ledger artifacts
    Given a session already has persisted compaction summary artifacts in SQLite
    And the gateway or agent process restarts
    When the next prompt is assembled for that session
    Then the working summary and distilled summary state are reconstructed from durable session records
    And the system does not depend on transient in-memory summary state surviving restart

  Scenario: Summary budget rules are model-aware
    Given different agents may use different foreground models with different context windows
    When the system evaluates summary reserve and prompt-facing summary budgets
    Then those budgets are derived from the active foreground model's usable context
    And the system does not assume one blanket summary budget for all agents

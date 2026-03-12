Feature: Discord system status explains provider-authoritative context pressure truthfully

  Scenario: Per-agent threshold is shown without blanket assumptions
    Given two agents can have different compaction threshold settings
    When the operator reads the Discord system status message
    Then the status message does not claim a single blanket threshold for all agents
    And the operator can determine the threshold that applies to each agent

  Scenario: System status shows the last provider-reported context usage
    Given an agent receives a provider response with total input token usage
    When the operator reads the Discord system status message
    Then the displayed context usage reflects the most recent provider-reported total input tokens
    And the status message does not substitute a local token estimate for that value

  Scenario: Last known provider-reported context usage persists across gateway restart
    Given an agent session has a valid persisted provider-reported usage snapshot
    And the gateway restarts before the next agent turn
    When the operator reads the Discord system status message
    Then the displayed context usage reflects the latest valid persisted provider-reported total input tokens
    And the status message does not reset that agent to zero solely because a new turn has not occurred yet

  Scenario: Normal compaction triggers from provider-reported usage
    Given the provider returns a response with total input token usage above the compaction threshold
    When the agent processes that response
    Then the agent starts structured compaction immediately
    And the compaction decision is based on the provider-reported total input tokens

  Scenario: Compaction does not trigger before provider response based on local estimates
    Given the agent has assembled a prompt for the next provider call
    When no provider response has yet been received
    Then the agent does not start normal compaction solely from a local token estimate

  Scenario: Provider overflow error triggers compaction recovery
    Given the provider rejects the prompt because the context window is too large
    When the agent receives that overflow error
    Then the agent starts structured compaction
    And retries once with the compacted session state

  Scenario: Operator can trigger a gateway restart from the system status message
    Given the Discord system status message is configured
    And the gateway supervisor is running
    When the operator clicks the restart button on the system status message
    Then nanobot requests a gateway restart via the supervisor restart path
    And the operator receives a visible confirmation in Discord

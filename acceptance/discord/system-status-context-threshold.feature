Feature: Discord system status explains context pressure truthfully

  Scenario: Per-agent threshold is shown without blanket assumptions
    Given two agents can have different compaction threshold settings
    When the operator reads the Discord system status message
    Then the status message does not claim a single blanket threshold for all agents
    And the operator can determine the threshold that applies to each agent

  Scenario: Current context is not conflated with compaction trigger state
    Given an agent has compacted previously
    When the operator reads the Discord system status message
    Then the displayed context usage reflects the agent's current assembled context snapshot
    And the status message does not imply that the displayed current token count was the compaction trigger value

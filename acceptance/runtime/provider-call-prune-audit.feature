Feature: Provider call and prompt-mutation records are auditable and replay-oriented

  Scenario: Provider responses are persisted as first-class provider call records
    Given the agent receives a provider response with usage metadata
    When the response is persisted
    Then a provider call record is written for that session
    And the record includes turn, iteration, model, finish reason, and provider-reported token counts
    And the record links to the prompt assembly snapshot used for that call
    And the record can be correlated to the messages produced by that call

  Scenario: Tool pruning mutations are persisted as first-class audit records
    Given tool results are pruned before a provider call
    When the pruning mutation is persisted
    Then a tool prune event is recorded for that session
    And each pruned result is queryable as a prune item record
    And the estimated savings and replacement mode are visible for audit

  Scenario: Prompt assembly state is replayable from normalized call-scoped records
    Given a provider call has an associated prompt assembly snapshot
    And the call-scoped dynamic prompt artifacts for that call were persisted
    When the operator requests replay of what the model saw
    Then the system can reconstruct the prompt from referenced session records and prompt artifacts
    And the reconstructed payload can be validated against a stored prompt hash

  Scenario: Dynamic prompt artifacts are pinned at call time for replay
    Given a provider call includes retrieved memory and turn-scoped context
    When the call is persisted
    Then the exact retrieved-memory payload used for that call is stored as a call-scoped snapshot
    And the exact rendered turn-context payload used for that call is stored as a call-scoped snapshot
    And later changes to retrieval logic or turn-context formatting do not invalidate replay of that call

  Scenario: Prompt pressure changes across turns can be explained
    Given provider call records, prune events, and compaction records exist for a session
    When the operator inspects prompt pressure across turns
    Then increases and decreases in provider-observed prompt size can be correlated to the relevant call-time mutations
    And the operator is not forced to infer prompt mutations from message content alone

  Scenario: Turn pressure can be approximated from provider call history
    Given a session has provider call records across multiple completed turns
    When the system evaluates continuity planning inputs
    Then it can derive approximate recent-turn prompt pressure from provider-observed call history
    And it can combine that with pruning and compaction records
    And it does not claim exact intrinsic per-message token attribution from provider totals alone

  Scenario: Operator can inspect provider-call audit records through the backend API
    Given provider call records exist for a session
    When the operator requests the session's provider-call list from the backend API
    Then the response includes call identity, turn/iteration, provider/model, finish reason, provider token totals, and linked assembly snapshot ids

  Scenario: Operator can inspect replay-oriented call details through the backend API
    Given a provider call has linked prompt assembly, retrieved-memory, and turn-context snapshots
    When the operator requests that provider call's detail from the backend API
    Then the response includes the provider call record and its linked replay-oriented snapshots
    And the response can be used to inspect what dynamic prompt artifacts were used for that call without reconstructing the entire prompt blob

  Scenario: Operator can inspect prompt-pressure history through the backend API
    Given provider calls, prune events, and compaction records exist for a session
    When the operator requests prompt-pressure history from the backend API
    Then the response correlates provider-observed prompt pressure with pruning and compaction events by turn/call
    And the response does not claim exact per-message token attribution from provider totals alone

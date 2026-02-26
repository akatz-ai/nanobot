---
schema_version: 1
id: router-consolidation
title: "Router/dispatcher consolidation \u2014 multi-channel routing policy chain"
status: active
priority: 3
created_at: '2026-02-26T20:42:08Z'
updated_at: '2026-02-26T20:42:09Z'
---

## Tasks
- [ ] router-consolidation.0 Create nanobot/routing/policy.py with RoutingPolicyChain
- [ ] router-consolidation.1 Implement CommandPolicy (/agents, /agent set/clear)
- [ ] router-consolidation.2 Implement ExplicitPrefixPolicy (@agentId prefix)
- [ ] router-consolidation.3 Implement StickySessionPolicy for thread-level agent stickiness
- [ ] router-consolidation.4 Implement StaticChannelBindingPolicy using merged_channel_bindings()
- [ ] router-consolidation.5 Add channel_bindings and slack_channels to AgentProfile schema
- [ ] router-consolidation.6 Patch AgentRouter to use policy chain
- [ ] router-consolidation.7 Tag outbound messages with agent_id + persona metadata

## Notes

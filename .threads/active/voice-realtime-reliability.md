---
schema_version: 1
id: voice-realtime-reliability
title: Voice Realtime Reliability and Latency
status: active
priority: 1
created_at: '2026-03-04T20:32:20Z'
updated_at: '2026-03-04T20:32:24Z'
---

## Tasks
- [ ] voice-realtime-reliability.0 Add end-to-end voice/realtime observability baseline (connect/startup/audio/timing/drop counters)
- [ ] voice-realtime-reliability.1 Fix startup capture race: subscribe receiver before/while session connects and avoid missing first utterance {deps=[voice-realtime-reliability.0]}
- [ ] voice-realtime-reliability.2 Harden receiver lifecycle: persistent owner subscription strategy and correct listener cleanup on stop/restart {deps=[voice-realtime-reliability.1]}
- [ ] voice-realtime-reliability.3 Rework idle/reconnect behavior to reduce first-turn latency after idle (keep-warm + bounded reconnect/backoff) {deps=[voice-realtime-reliability.2]}
- [ ] voice-realtime-reliability.4 Tune and externalize VAD/noise settings for driving/walking environments (profiled defaults) {deps=[voice-realtime-reliability.3]}
- [ ] voice-realtime-reliability.5 Implement barge-in interruption: cancel/truncate assistant response when user speech starts {deps=[voice-realtime-reliability.4]}
- [ ] voice-realtime-reliability.6 Reduce hot-path logging and disable verbose voice debug by default {deps=[voice-realtime-reliability.5]}
- [ ] voice-realtime-reliability.7 Optimize audio conversion pipeline and add backpressure guards for sustained sessions {deps=[voice-realtime-reliability.6]}
- [ ] voice-realtime-reliability.8 Add integration tests + soak checks for startup latency and missed-utterance regressions {deps=[voice-realtime-reliability.7]}
- [ ] voice-realtime-reliability.9 Document runbook, KPIs, and rollout gates for production enablement {deps=[voice-realtime-reliability.8]}

## Notes

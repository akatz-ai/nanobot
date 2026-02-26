---
schema_version: 1
id: security-hardening
title: "Security hardening \u2014 sandbox exec, SSRF protection, file tool restrictions"
status: active
priority: 2
created_at: '2026-02-26T20:42:21Z'
updated_at: '2026-02-26T20:42:22Z'
---

## Tasks
- [ ] security-hardening.0 Add SSRF protection to web_fetch: block private IP ranges, localhost, link-local
- [ ] security-hardening.1 Enforce allowed_dir by default in file tools (not just when explicitly set)
- [ ] security-hardening.2 Evaluate containerized exec: run shell commands in restricted subprocess or namespace
- [ ] security-hardening.3 Audit MCP tool trust boundary — document what MCP servers can access

## Notes

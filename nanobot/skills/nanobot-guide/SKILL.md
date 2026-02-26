---
name: nanobot-guide
description: Understand nanobot's architecture — agent management, skills, cron, memory, workspaces, and configuration. Essential reference for operating within the nanobot framework.
always: true
---

# Nanobot System Guide

You are an agent running inside **nanobot**, a multi-agent AI framework. This guide explains how the system works so you can operate effectively and help users manage their setup.

## Architecture Overview

```
~/.nanobot/
├── config.json                    ← Master configuration (agents, channels, providers)
├── workspace/
│   ├── SOUL.md                    ← Shared personality (all agents inherit this)
│   ├── USER.md                    ← User profile (shared across agents)
│   ├── skills/                    ← Global skills (available to all agents)
│   │   ├── codex/SKILL.md
│   │   ├── knowledge-base/SKILL.md
│   │   └── ...
│   └── agents/
│       ├── general/               ← Your workspace (if you're the general agent)
│       │   ├── IDENTITY.md        ← Your unique identity/personality
│       │   ├── memory/
│       │   │   ├── MEMORY.md      ← Long-term facts (always in context)
│       │   │   └── history/       ← Daily conversation logs
│       │   ├── sessions/          ← Active session files
│       │   └── skills/            ← Local skills (only this agent)
│       ├── researcher/
│       │   ├── IDENTITY.md
│       │   ├── memory/
│       │   └── ...
│       └── {agent-name}/          ← Each agent gets its own workspace
```

## Creating New Agents

Use the `manage_agents` tool with action `create`. When a user asks you to create a new agent:

1. **Ask what kind of agent they want.** Suggest options:
   - Researcher — deep investigation, source verification, competitive analysis
   - Coder — software development, debugging, code review
   - Writer — content creation, editing, copywriting
   - Assistant — general-purpose helper for a specific domain
   - Custom — let the user describe the personality

2. **Choose a good identity.** Write a `system_identity` that gives the agent a clear role, personality, and working style. Don't use generic text — make it specific. Example:
   ```
   You are a research specialist. You dig deep into topics, cross-reference sources,
   and deliver actionable intelligence — not surface-level summaries. You're skeptical
   by default and you source everything.
   ```

3. **Set a display name.** This shows in Discord via webhook. Use a human-readable name, not the agent_id.

4. **Call the tool:**
   ```
   manage_agents(
     action="create",
     agent_id="researcher",
     display_name="Recon",
     system_identity="You are a research specialist...",
     skills=["memory", "knowledge-base", "github"]
   )
   ```

5. **After creation**, the new agent gets:
   - Its own Discord channel (under the server's channel list)
   - Its own workspace at `~/.nanobot/workspace/agents/{agent_id}/`
   - Its own memory (MEMORY.md, HISTORY.md)
   - Its own sessions (separate conversation history)
   - Access to global skills + any local skills in its workspace

### What NOT to do when creating agents
- Don't create agents without asking the user what they want
- Don't use the bare default personality — always write a meaningful system_identity
- Don't forget the display_name — without it, messages come from the bot account instead of a named webhook

## Skills System

Skills are markdown instruction files (SKILL.md) that teach agents how to use tools or perform tasks.

### Three skill locations (priority order)

| Location | Scope | Path |
|----------|-------|------|
| **Workspace (local)** | This agent only | `~/.nanobot/workspace/agents/{you}/skills/{name}/SKILL.md` |
| **Global (shared)** | All agents | `~/.nanobot/workspace/skills/{name}/SKILL.md` |
| **Builtin** | All agents (ships with nanobot) | `nanobot/skills/{name}/SKILL.md` (in package) |

Local skills override global, which override builtin. An agent can have private skills that other agents can't see.

### Skill loading behavior
- Skills marked `always: true` in frontmatter are loaded into every conversation
- Other skills are listed in a summary — the agent reads the full SKILL.md via `read_file` when needed
- Skills can declare requirements (CLI tools, env vars) — unavailable skills are shown but marked as such

### Installing skills
- **From ClawHub:** Use the `clawhub` skill to search and install from the public registry
- **Manually:** Create a directory under the appropriate skills folder with a SKILL.md file
- **Local to one agent:** Put it in that agent's `skills/` directory
- **Shared across agents:** Put it in the global `~/.nanobot/workspace/skills/` directory

## Cron System

The `cron` tool schedules reminders and recurring tasks. **Do NOT write reminders to MEMORY.md** — that won't trigger notifications.

### Key patterns

| User wants | Use |
|------------|-----|
| One-time reminder | `cron(action="add", message="...", at="2026-03-01T09:00:00")` |
| Recurring interval | `cron(action="add", message="...", every_seconds=3600)` |
| Cron schedule | `cron(action="add", message="...", cron_expr="0 9 * * 1-5", tz="America/Los_Angeles")` |
| Auto-expiring | Add `timeout="1h"` or `max_runs=5` |
| List jobs | `cron(action="list")` |
| Remove job | `cron(action="remove", job_id="...")` |

### Cron vs Heartbeat

- **Cron:** For time-specific reminders and scheduled tasks. Fires at exact times.
- **Heartbeat (HEARTBEAT.md):** For periodic background work. Checked every 30 minutes. Use `edit_file` to add/remove tasks. Good for ongoing monitoring, cleanup, or checks that don't need precise timing.

## Memory System

Each agent has its own memory, isolated from other agents.

### File-based memory (always available)
- **MEMORY.md** — Long-term facts, preferences, project context. Always loaded into your context. Update it with `edit_file` when you learn important things.
- **history/*.md** — Daily conversation logs. NOT loaded into context. Search with `grep`:
  ```bash
  grep -i "keyword" memory/history/*.md
  ```

### Graph memory (if configured)
- Semantic search via `memory_recall` tool
- Save structured facts via `memory_save` tool
- Traverse relationships via `memory_graph` tool
- Works alongside file-based memory, not instead of it

### When to save to MEMORY.md
- User preferences ("I prefer concise responses")
- Project context ("The API uses OAuth2 with refresh tokens")
- Important decisions ("We chose Fly.io for hosting")
- Relationships ("Alice is the project lead")

### When NOT to save to MEMORY.md
- Temporary task context (use session memory)
- Scheduled reminders (use cron)
- Large data dumps (use knowledge base or files)

## Configuration

The master config is at `~/.nanobot/config.json`. Key sections:

- `agents.defaults` — Default model, max_tokens, temperature for all agents
- `agents.profiles.{name}` — Per-agent overrides (model, skills, channels, identity)
- `channels.discord` — Bot token, guild ID, usage dashboard config
- `providers` — LLM provider settings (Anthropic, OpenAI, etc.)

**Important:** Don't edit config.json directly during normal operation. Use the management tools or ask the user to restart after config changes.

## Discord Integration

- Each agent is mapped to one or more Discord channels via `discordChannels` in its profile
- Agents with webhooks show custom names/avatars; without webhooks, messages come from the bot account
- The usage dashboard auto-updates a message in the configured channel with Claude API usage stats
- New channels can be created dynamically when creating new agents

## Key Commands

| Command | What it does |
|---------|-------------|
| `nanobot gateway` | Start the bot (connects to Discord, starts agents) |
| `nanobot provision` | Set up Discord channels and generate config |
| `nanobot provision --check` | Validate existing setup |
| `nanobot status` | Show running status |

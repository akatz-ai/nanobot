# Nanobot Voice Agent

Discord voice sidecar for nanobot that connects to OpenAI Realtime API.

## What it does

- Joins a configured voice channel when the configured owner user joins
- Streams owner audio from Discord to OpenAI Realtime (`gpt-realtime-mini`)
- Plays OpenAI voice responses back into Discord
- Exposes nanobot-oriented tools (channel read/send with confirmation gate, system status, usage, list agents)

## Setup

1. Use Node 22+
2. Install dependencies:

```bash
npm install
```

3. Copy `.env.example` to `.env` and fill values
4. Build and run:

```bash
npm run build
npm start
```

## Runtime behavior

- Watches `DISCORD_OWNER_USER_ID` and `DISCORD_VOICE_CHANNEL_ID`
- Joins when owner enters the channel, leaves when owner exits
- Streams owner speech to OpenAI Realtime (`input_audio_buffer.append`)
- Plays `response.output_audio.delta` back to Discord voice

## Tools (Phase 2)

- `read_agent_channel`
- `send_message_to_agent` (queue only)
- `confirm_action` (executes queued send)
- `cancel_action`
- `check_system_status`
- `check_usage`
- `list_agents`
- `search_knowledge_base` (placeholder: not implemented)
- `search_agent_memory` (placeholder: not implemented)

## Notes

- `send_message_to_agent` never sends directly. It queues a pending action.
- `confirm_action` is required to execute queued sends.
- Pending actions expire after 60 seconds.

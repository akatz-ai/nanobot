---
schema_version: 1
id: discord-forward-support
title: 'Discord: Support forwarded messages, embeds, and rich content'
status: active
priority: 3
created_at: '2026-02-26T05:07:28Z'
updated_at: '2026-02-26T05:07:47Z'
---

## Tasks
- [ ] discord-forward-support.0 Extract content from message_snapshots (forwarded messages)
- [ ] discord-forward-support.1 Extract text from embeds (title, description, fields)
- [ ] discord-forward-support.2 Download images from embed.image/thumbnail URLs
- [ ] discord-forward-support.3 Handle forwarded message attachments
- [ ] discord-forward-support.4 Test with real forwarded messages in Discord {deps=[discord-forward-support.0,discord-forward-support.1,discord-forward-support.2,discord-forward-support.3]}

## Notes
## Research (2026-02-25)

### Discord Forwarding API
- Forwarded messages use `message_reference.type = 1` (FORWARD)
- Content lives in `message_snapshots[].message` — a partial message object
- Snapshot fields: type, content, embeds, attachments, timestamp, edited_timestamp, flags, mentions, mention_roles, stickers, sticker_items, components
- Message has HAS_SNAPSHOT flag (bit 14, value 16384)
- The forwarding message's own `content` field is EMPTY
- Nested forwards limited to depth 1

### Embeds
- `embeds[]` array with: title, description, url, fields[], author, footer, image, thumbnail
- Used for link previews, rich content, bot embeds

### Current State
- `discord.py _handle_message_create` only processes `content` and `attachments`
- `embeds` and `message_snapshots` are silently dropped
- We have MESSAGE_CONTENT intent (required) already enabled
- Estimated ~30-40 lines of additive code in `_handle_message_create`
- No architectural changes needed — just enrich content_parts and media_paths

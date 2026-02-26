"""Discord channel implementation using Discord Gateway websocket."""

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import websockets
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig


DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024  # 25MB (Discord limit)
MAX_ATTACHMENTS_PER_MSG = 10  # Discord limit
MAX_MESSAGE_LEN = 2000  # Discord message character limit


def _split_message(content: str, max_len: int = MAX_MESSAGE_LEN) -> list[str]:
    """Split content into chunks within max_len, preferring line breaks."""
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        pos = cut.rfind('\n')
        if pos <= 0:
            pos = cut.rfind(' ')
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


class DiscordChannel(BaseChannel):
    """Discord channel using Gateway websocket."""

    name = "discord"

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._seq: int | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._http: httpx.AsyncClient | None = None
        # Webhook registry: channel_id -> {webhook_url, display_name, avatar_url}
        self._webhooks: dict[str, dict[str, str | None]] = {}

    async def start(self) -> None:
        """Start the Discord gateway connection."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)

        while self._running:
            try:
                logger.info("Connecting to Discord gateway...")
                async with websockets.connect(self.config.gateway_url) as ws:
                    self._ws = ws
                    await self._gateway_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Discord gateway error: {}", e)
                if self._running:
                    logger.info("Reconnecting to Discord gateway in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the Discord channel."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._http:
            await self._http.aclose()
            self._http = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord REST API or webhook."""
        if not self._http:
            logger.warning("Discord HTTP client not initialized")
            return

        # Check if this channel has a registered webhook
        webhook_info = self._webhooks.get(msg.chat_id)
        if webhook_info and webhook_info.get("webhook_url"):
            await self._send_via_webhook(msg, webhook_info)
            return

        url = f"{DISCORD_API_BASE}/channels/{msg.chat_id}/messages"
        headers = {"Authorization": f"Bot {self.config.token}"}

        try:
            # Send media files first (batched up to 10 per message)
            media_files = self._resolve_media(msg.media)
            await self._send_media_batches(url, headers, media_files)

            # Send text content
            chunks = _split_message(msg.content or "")
            for i, chunk in enumerate(chunks):
                payload: dict[str, Any] = {"content": chunk}

                # Only set reply reference on the first chunk
                if i == 0 and msg.reply_to:
                    payload["message_reference"] = {"message_id": msg.reply_to}
                    payload["allowed_mentions"] = {"replied_user": False}

                if not await self._send_payload(url, headers, payload):
                    break  # Abort remaining chunks on failure
        finally:
            await self._stop_typing(msg.chat_id)

    async def _send_via_webhook(self, msg: OutboundMessage, webhook_info: dict[str, str | None]) -> None:
        """Send a message via Discord webhook with custom name/avatar."""
        webhook_url = webhook_info["webhook_url"]

        # Base payload fields for webhook
        base_payload: dict[str, Any] = {}
        if webhook_info.get("display_name"):
            base_payload["username"] = webhook_info["display_name"]
        if webhook_info.get("avatar_url"):
            base_payload["avatar_url"] = webhook_info["avatar_url"]

        try:
            # Send media files first
            media_files = self._resolve_media(msg.media)
            await self._send_media_batches(webhook_url, {}, media_files, extra_payload=base_payload)

            # Send text content
            chunks = _split_message(msg.content or "")
            for chunk in chunks:
                payload = {**base_payload, "content": chunk}

                if not await self._send_payload(webhook_url, {}, payload):
                    break
        finally:
            await self._stop_typing(msg.chat_id)

    @staticmethod
    def _resolve_media(media: list[str] | None) -> list[Path]:
        """Resolve media paths to Path objects, filtering out missing/oversized files."""
        if not media:
            return []
        resolved: list[Path] = []
        for item in media:
            p = Path(item)
            if not p.is_file():
                logger.warning("Media file not found, skipping: {}", item)
                continue
            if p.stat().st_size > MAX_ATTACHMENT_BYTES:
                logger.warning("Media file too large (>25MB), skipping: {}", item)
                continue
            resolved.append(p)
        return resolved

    async def _send_media_batches(
        self,
        url: str,
        headers: dict[str, str],
        files: list[Path],
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        """Upload media files in batches of up to 10 per Discord message."""
        if not files:
            return
        for i in range(0, len(files), MAX_ATTACHMENTS_PER_MSG):
            batch = files[i : i + MAX_ATTACHMENTS_PER_MSG]
            await self._send_multipart(url, headers, batch, extra_payload)

    async def _send_multipart(
        self,
        url: str,
        headers: dict[str, str],
        files: list[Path],
        extra_payload: dict[str, Any] | None = None,
    ) -> bool:
        """Send files as multipart/form-data to Discord. Returns True on success."""
        if not self._http or not files:
            return False

        for attempt in range(3):
            try:
                # Build multipart fields
                form_files: dict[str, Any] = {}
                attachments_json: list[dict[str, Any]] = []

                for idx, fp in enumerate(files):
                    mime = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
                    form_files[f"files[{idx}]"] = (fp.name, fp.read_bytes(), mime)
                    attachments_json.append({"id": idx, "filename": fp.name})

                payload: dict[str, Any] = {"attachments": attachments_json}
                if extra_payload:
                    payload.update(extra_payload)

                form_files["payload_json"] = (
                    None,
                    json.dumps(payload),
                    "application/json",
                )

                response = await self._http.post(url, headers=headers, files=form_files)
                if response.status_code == 429:
                    data = response.json()
                    retry_after = float(data.get("retry_after", 1.0))
                    logger.warning("Discord rate limited, retrying in {}s", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                names = [fp.name for fp in files]
                logger.info("Sent {} media file(s) to Discord: {}", len(files), names)
                return True
            except Exception as e:
                if attempt == 2:
                    logger.error("Error sending Discord media: {}", e)
                else:
                    await asyncio.sleep(1)
        return False

    async def _send_payload(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> bool:
        """Send a single Discord API payload with retry on rate-limit. Returns True on success."""
        for attempt in range(3):
            try:
                response = await self._http.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    data = response.json()
                    retry_after = float(data.get("retry_after", 1.0))
                    logger.warning("Discord rate limited, retrying in {}s", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                return True
            except Exception as e:
                if attempt == 2:
                    logger.error("Error sending Discord message: {}", e)
                else:
                    await asyncio.sleep(1)
        return False

    async def _gateway_loop(self) -> None:
        """Main gateway loop: identify, heartbeat, dispatch events."""
        if not self._ws:
            return

        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from Discord gateway: {}", raw[:100])
                continue

            op = data.get("op")
            event_type = data.get("t")
            seq = data.get("s")
            payload = data.get("d")

            if seq is not None:
                self._seq = seq

            if op == 10:
                # HELLO: start heartbeat and identify
                interval_ms = payload.get("heartbeat_interval", 45000)
                await self._start_heartbeat(interval_ms / 1000)
                await self._identify()
            elif op == 0 and event_type == "READY":
                logger.info("Discord gateway READY")
            elif op == 0 and event_type == "MESSAGE_CREATE":
                await self._handle_message_create(payload)
            elif op == 7:
                # RECONNECT: exit loop to reconnect
                logger.info("Discord gateway requested reconnect")
                break
            elif op == 9:
                # INVALID_SESSION: reconnect
                logger.warning("Discord gateway invalid session")
                break

    async def _identify(self) -> None:
        """Send IDENTIFY payload."""
        if not self._ws:
            return

        identify = {
            "op": 2,
            "d": {
                "token": self.config.token,
                "intents": self.config.intents,
                "properties": {
                    "os": "nanobot",
                    "browser": "nanobot",
                    "device": "nanobot",
                },
            },
        }
        await self._ws.send(json.dumps(identify))

    async def _start_heartbeat(self, interval_s: float) -> None:
        """Start or restart the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        async def heartbeat_loop() -> None:
            while self._running and self._ws:
                payload = {"op": 1, "d": self._seq}
                try:
                    await self._ws.send(json.dumps(payload))
                except Exception as e:
                    logger.warning("Discord heartbeat failed: {}", e)
                    break
                await asyncio.sleep(interval_s)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def _handle_message_create(self, payload: dict[str, Any]) -> None:
        """Handle incoming Discord messages."""
        author = payload.get("author") or {}
        if author.get("bot"):
            return

        sender_id = str(author.get("id", ""))
        channel_id = str(payload.get("channel_id", ""))
        content = payload.get("content") or ""

        if not sender_id or not channel_id:
            return

        if not self.is_allowed(sender_id):
            return

        content_parts = [content] if content else []
        media_paths: list[str] = []
        media_dir = Path.home() / ".nanobot" / "media"

        for attachment in payload.get("attachments") or []:
            url = attachment.get("url")
            filename = attachment.get("filename") or "attachment"
            size = attachment.get("size") or 0
            if not url or not self._http:
                continue
            if size and size > MAX_ATTACHMENT_BYTES:
                content_parts.append(f"[attachment: {filename} - too large]")
                continue
            try:
                media_dir.mkdir(parents=True, exist_ok=True)
                file_path = media_dir / f"{attachment.get('id', 'file')}_{filename.replace('/', '_')}"
                resp = await self._http.get(url)
                resp.raise_for_status()
                file_path.write_bytes(resp.content)
                media_paths.append(str(file_path))
                content_parts.append(f"[attachment: {file_path}]")
            except Exception as e:
                logger.warning("Failed to download Discord attachment: {}", e)
                content_parts.append(f"[attachment: {filename} - download failed]")

        reply_to = (payload.get("referenced_message") or {}).get("id")

        await self._start_typing(channel_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            content="\n".join(p for p in content_parts if p) or "[empty message]",
            media=media_paths,
            metadata={
                "message_id": str(payload.get("id", "")),
                "guild_id": payload.get("guild_id"),
                "reply_to": reply_to,
            },
        )

    async def _start_typing(self, channel_id: str) -> None:
        """Start periodic typing indicator for a channel."""
        await self._stop_typing(channel_id)

        async def typing_loop() -> None:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/typing"
            headers = {"Authorization": f"Bot {self.config.token}"}
            while self._running:
                try:
                    await self._http.post(url, headers=headers)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.debug("Discord typing indicator failed for {}: {}", channel_id, e)
                    return
                await asyncio.sleep(8)

        self._typing_tasks[channel_id] = asyncio.create_task(typing_loop())

    async def _stop_typing(self, channel_id: str) -> None:
        """Stop typing indicator for a channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task:
            task.cancel()

    # ------------------------------------------------------------------
    # Guild management REST API (for dynamic channel creation)
    # ------------------------------------------------------------------

    async def create_guild_channel(
        self,
        guild_id: str,
        name: str,
        topic: str | None = None,
        category_id: str | None = None,
        channel_type: int = 0,
    ) -> str | None:
        """Create a guild channel. Defaults to type=0 (text). Returns channel ID or None."""
        owns_client = self._http is None
        client = self._http or httpx.AsyncClient(timeout=30.0)

        url = f"{DISCORD_API_BASE}/guilds/{guild_id}/channels"
        headers = {"Authorization": f"Bot {self.config.token}"}
        payload: dict[str, Any] = {"name": name, "type": channel_type}
        if topic and channel_type == 0:
            payload["topic"] = topic
        if category_id and channel_type == 0:
            payload["parent_id"] = category_id

        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            channel_id = str(data["id"])
            logger.info("Created Discord channel #{} ({})", name, channel_id)
            return channel_id
        except Exception as e:
            logger.error("Failed to create Discord channel '{}': {}", name, e)
            return None
        finally:
            if owns_client:
                await client.aclose()

    async def create_channel_webhook(
        self,
        channel_id: str,
        name: str = "Agent",
        avatar_url: str | None = None,
    ) -> str | None:
        """Create a webhook for a channel. Returns the webhook URL or None on failure."""
        owns_client = self._http is None
        client = self._http or httpx.AsyncClient(timeout=30.0)

        url = f"{DISCORD_API_BASE}/channels/{channel_id}/webhooks"
        headers = {"Authorization": f"Bot {self.config.token}"}
        payload: dict[str, Any] = {"name": name}
        if avatar_url:
            logger.debug("Discord webhook avatar_url ignored (URL-to-avatar upload not implemented)")

        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            webhook_url = f"{DISCORD_API_BASE}/webhooks/{data['id']}/{data['token']}"
            logger.info("Created webhook for channel {} (name={})", channel_id, name)
            return webhook_url
        except Exception as e:
            logger.error("Failed to create webhook for channel {}: {}", channel_id, e)
            return None
        finally:
            if owns_client:
                await client.aclose()

    def register_webhook(
        self,
        channel_id: str,
        webhook_url: str,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> None:
        """Register a webhook for a channel so outbound messages use it."""
        self._webhooks[channel_id] = {
            "webhook_url": webhook_url,
            "display_name": display_name,
            "avatar_url": avatar_url,
        }
        logger.info("Registered webhook for channel {} (name={})", channel_id, display_name)

    async def list_guild_channels(self, guild_id: str) -> list[dict[str, Any]]:
        """List all channels in a guild."""
        owns_client = self._http is None
        client = self._http or httpx.AsyncClient(timeout=30.0)

        url = f"{DISCORD_API_BASE}/guilds/{guild_id}/channels"
        headers = {"Authorization": f"Bot {self.config.token}"}

        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Failed to list Discord guild channels: {}", e)
            return []
        finally:
            if owns_client:
                await client.aclose()

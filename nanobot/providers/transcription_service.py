"""Unified voice transcription service with provider fallback."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

AUDIO_EXTENSIONS = {".ogg", ".mp3", ".m4a", ".wav", ".aac", ".opus", ".flac", ".webm"}


def is_audio_file(path: str | Path) -> bool:
    """Check if a file path has an audio extension."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


class TranscriptionService:
    """Tries Groq first, then falls back to OpenAI."""

    def __init__(self, groq_api_key: str = "", openai_api_key: str = ""):
        self.groq_api_key = groq_api_key
        self.openai_api_key = openai_api_key

    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file. Returns an empty string on failure."""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        if self.groq_api_key:
            try:
                from nanobot.providers.transcription import GroqTranscriptionProvider

                provider = GroqTranscriptionProvider(api_key=self.groq_api_key)
                result = await provider.transcribe(path)
                if result:
                    logger.info("Transcribed via Groq: {}...", result[:50])
                    return result
            except Exception as e:
                logger.warning("Groq transcription failed, trying OpenAI: {}", e)

        if self.openai_api_key:
            try:
                from nanobot.providers.openai_transcription import (
                    OpenAITranscriptionProvider,
                )

                provider = OpenAITranscriptionProvider(api_key=self.openai_api_key)
                result = await provider.transcribe(path)
                if result:
                    logger.info("Transcribed via OpenAI: {}...", result[:50])
                    return result
            except Exception as e:
                logger.warning("OpenAI transcription failed: {}", e)

        logger.warning("No transcription provider available for {}", file_path)
        return ""

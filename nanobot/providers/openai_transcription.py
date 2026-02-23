"""Voice transcription provider using OpenAI Whisper."""

import os
from pathlib import Path

import httpx
from loguru import logger


class OpenAITranscriptionProvider:
    """
    Voice transcription provider using OpenAI's Whisper API.
    """
    
    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
    
    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using OpenAI Whisper.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("OpenAI API key not configured for transcription")
            return ""
        
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        
        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f, "audio/ogg"),
                    }
                    data = {
                        "model": self.model,
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }
                    
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=60.0
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    return result.get("text", "")
                    
        except Exception as e:
            logger.error("OpenAI transcription error: {}", e)
            return ""
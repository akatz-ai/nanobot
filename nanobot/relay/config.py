"""Relay client configuration models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class RelayClientConfig(BaseModel):
    """Outbound relay tunnel configuration."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    enabled: bool = False
    url: str = ""
    token: str = ""
    instance_id: str = ""
    reconnect_interval_s: int = 5
    max_reconnect_interval_s: int = 60
    request_timeout_s: int = 30

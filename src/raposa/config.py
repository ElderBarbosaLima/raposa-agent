from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAPOSA_", extra="ignore")

    model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    max_tool_rounds: int = 6
    db_path: Path = Path("data/raposa.db")
    football_data_api_key: str | None = Field(
        default=None, validation_alias="FOOTBALL_DATA_API_KEY"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Application configuration leveraging environment variables with sensible defaults."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration values for the backend services."""

    app_name: str = Field(default="Repo RAG Analyst")
    api_prefix: str = Field(default="/api")
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Ingestion settings
    workspace_dir: Path = Field(default=Path(".cache/workspace"))
    github_token: Optional[str] = Field(default=None, description="GitHub personal access token if using API")

    # Embedding / LLM settings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = Field(default="gemini-2.0-flash")
    google_api_key: Optional[str] = Field(default=None)

    # RAG settings
    chunk_size: int = Field(default=750)
    chunk_overlap: int = Field(default=150)
    max_files: int = Field(default=500)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Cache and return application settings."""

    settings = Settings()
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    return settings

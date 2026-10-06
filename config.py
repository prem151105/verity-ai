"""
Verity — Application Configuration
All settings loaded from environment variables / .env file.
Updated to Pydantic V2 / pydantic-settings V2 syntax.
"""

from typing import Literal
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    # SEC EDGAR
    sec_user_agent: str = Field(
        default="Verity/1.0 portfolio@example.com", alias="SEC_USER_AGENT"
    )
    edgar_request_delay: float = Field(default=0.1, alias="EDGAR_REQUEST_DELAY")

    # News
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    # Storage
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    audit_log_dir: str = Field(default="./audit_logs", alias="AUDIT_LOG_DIR")
    retrieval_backend: Literal['bm25', 'gemini'] = 'bm25'
    research_mode: Literal['fast', 'deep'] = 'fast'

    # Agent config
    verifier_max_retries: int = Field(default=2, ge=1, le=5, alias="VERIFIER_MAX_RETRIES")

    verification_backend: Literal["rules", "torch", "onnx"] = "rules"
    verification_remote_budget: int = Field(default=24, ge=0, le=100)

    # Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()

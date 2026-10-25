"""Settings read from the environment or the repo-root .env file; a relative DATA_DIR is resolved against the repo root."""
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]  # arxiv-paper-rag/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    # auto: multilingual-e5 embeddings when the "embed" extra is installed, else BM25 only. off: always BM25 only.
    embed: Literal["auto", "off"] = "auto"
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    # declared so a key in .env reaches the SDK: pydantic-settings drops .env entries it has no field for
    anthropic_api_key: SecretStr | None = None
    # arXiv asks for at most one request every 3 seconds.
    arxiv_delay_s: float = 3.0

    @field_validator("data_dir")
    @classmethod
    def _anchor_to_root(cls, v: Path) -> Path:
        # the server may be started from another directory, so a relative path must not depend on the working directory
        return v if v.is_absolute() else ROOT / v


settings = Settings()

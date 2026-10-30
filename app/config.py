"""Settings read from the environment or the repo-root .env file; a relative DATA_DIR is resolved against the repo root."""
from pathlib import Path
from typing import Annotated, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]  # arxiv-paper-rag/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    # auto: multilingual-e5 embeddings when the "embed" extra is installed, else BM25 only. off: always BM25 only.
    embed: Literal["auto", "off"] = "auto"
    # auto: rerank the first RERANK_DEPTH candidates with a cross-encoder (needs the "embed" extra). Off until measured.
    rerank: Literal["auto", "off"] = "off"
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual, like the embedder
    rerank_depth: int = 30
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    # claim-support check: a second call that grades each cited sentence against its passage
    verify: bool = True
    verify_model: str = ""  # empty: the same model as the answers
    verify_effort: str = "low"
    # agentic retrieval: the model runs follow-up searches before answering (off by default until measured)
    agentic: bool = False
    agent_rounds: int = 3  # model turns that may call the search tool
    # declared so a key in .env reaches the SDK: pydantic-settings drops .env entries it has no field for
    anthropic_api_key: SecretStr | None = None
    # arXiv asks for at most one request every 3 seconds.
    arxiv_delay_s: float = 3.0

    # Host headers the web app answers to (DNS-rebinding defence); ALLOWED_HOSTS is comma-separated.
    allowed_hosts: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1", "[::1]"]
    # when set, POST /api/ask needs this value in the X-API-Token header
    api_token: SecretStr | None = None

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, v):
        return [h.strip() for h in v.split(",") if h.strip()] if isinstance(v, str) else v

    @field_validator("api_token", mode="before")
    @classmethod
    def _empty_token_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("data_dir")
    @classmethod
    def _anchor_to_root(cls, v: Path) -> Path:
        # the server may be started from another directory, so a relative path must not depend on the working directory
        return v if v.is_absolute() else ROOT / v


settings = Settings()

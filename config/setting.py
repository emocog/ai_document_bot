# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Settings(BaseSettings):
    # ▶ Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "gdrive_docs0"

    # ▶ 임베딩 / LLM 서버
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_API_BASE: str
    EMBED_API_KEY: str
    LLM_MODEL: str = "Qwen/Qwen3-32B"
    LLM_API_BASE: str
    LLM_API_KEY: str

    # ▶ 검색
    TOP_K: int = 8

    # ▶ 구글 Drive
    SERVICE_ACCOUNT_KEY: str = "service_account_key.json"

    # ▶ API 보호 (선택)
    API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = _Settings()  # 외부 모듈에서 import

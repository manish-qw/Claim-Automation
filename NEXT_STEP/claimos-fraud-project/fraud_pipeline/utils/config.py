"""
Application configuration — environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

# Absolute path to .env — works regardless of which directory uvicorn starts from
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────
    APP_NAME: str = "ClaimOS Fraud Intelligence Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── PostgreSQL ────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "claimos_fraud"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def POSTGRES_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def POSTGRES_URL_SYNC(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Neo4j ─────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_secret"

    # ── Redis ─────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── LLM ──────────────────────────────────────────
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "gemini"          # gemini | openai
    LLM_MODEL_GEMINI: str = "gemini-2.0-flash"
    LLM_MODEL_OPENAI: str = "gpt-4o"

    # ── Fraud Thresholds ─────────────────────────────
    FRAUD_HIGH_THRESHOLD: float = 0.75
    FRAUD_MEDIUM_THRESHOLD: float = 0.50
    EARLY_CLAIM_DAYS_THRESHOLD: int = 180
    TRUST_HUMAN_REVIEW_THRESHOLD: float = 0.70
    OCR_LOW_CONFIDENCE_THRESHOLD: float = 0.50

    # ── Sentence Transformers ─────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    NON_DISCLOSURE_SIMILARITY_THRESHOLD: float = 0.35  # below = contradiction

    # ── App ───────────────────────────────────────────
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = _ENV_FILE
        case_sensitive = True
        extra = "ignore"

settings = Settings()

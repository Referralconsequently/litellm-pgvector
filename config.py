from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBFieldsConfig(BaseModel):
    """Database column names for embedding rows."""

    id_field: str = "id"
    vector_store_id_field: str = "vector_store_id"
    content_field: str = "content"
    embedding_field: str = "embedding"
    metadata_field: str = "metadata"
    created_at_field: str = "created_at"


class EmbeddingConfig(BaseModel):
    """OpenAI-compatible embedding route served through the LiteLLM proxy."""

    model: str = "openai/local-pgvector-embedding"
    base_url: str = "http://127.0.0.1:4000"
    api_key: str = "your-api-key-here"
    dimensions: int = 1536


class Settings(BaseSettings):
    """Runtime settings for the local LiteLLM pgvector sidecar.

    Environment variables are loaded with pydantic-settings. Nested values use
    double underscores, for example EMBEDDING__MODEL or DB_FIELDS__ID_FIELD.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/litellm_pgvector"
    server_api_key: str = "your-api-key-here"
    port: int = 8000
    host: str = "127.0.0.1"

    table_names: dict[str, str] = Field(
        default_factory=lambda: {
            "vector_stores": "vector_stores",
            "embeddings": "embeddings",
        }
    )
    db_fields: DBFieldsConfig = Field(default_factory=DBFieldsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


settings = Settings()

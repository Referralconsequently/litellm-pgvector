from typing import Any

from pydantic import BaseModel


class VectorStoreCreateRequest(BaseModel):
    name: str
    file_ids: list[str] | None = None
    expires_after: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class VectorStoreResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    usage_bytes: int
    file_counts: dict[str, int]
    status: str
    expires_after: dict[str, Any] | None = None
    expires_at: int | None = None
    last_active_at: int | None = None
    metadata: dict[str, Any] | None = None


class VectorStoreSearchRequest(BaseModel):
    query: str
    limit: int | None = 20
    filters: dict[str, Any] | None = None
    return_metadata: bool | None = True


class ContentChunk(BaseModel):
    type: str = "text"
    text: str


class SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: dict[str, Any] | None = None
    content: list[ContentChunk]


class VectorStoreSearchResponse(BaseModel):
    object: str = "vector_store.search_results.page"
    search_query: str
    data: list[SearchResult]
    has_more: bool = False
    next_page: str | None = None


class EmbeddingCreateRequest(BaseModel):
    content: str
    embedding: list[float]
    metadata: dict[str, Any] | None = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "embedding"
    vector_store_id: str
    content: str
    metadata: dict[str, Any] | None = None
    created_at: int


class EmbeddingBatchCreateRequest(BaseModel):
    embeddings: list[EmbeddingCreateRequest]


class EmbeddingBatchCreateResponse(BaseModel):
    object: str = "embedding.batch"
    data: list[EmbeddingResponse]
    created: int


class VectorStoreListResponse(BaseModel):
    object: str = "list"
    data: list[VectorStoreResponse]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False

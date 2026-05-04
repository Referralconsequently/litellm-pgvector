from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from config import EmbeddingConfig, settings
from langfuse import observe
from litellm import aembedding


if TYPE_CHECKING:
    from litellm.types.utils import EmbeddingResponse


_log = logging.getLogger(__name__)


def _raise_embedding_dimension_error(
    *,
    expected_dimensions: int,
    actual_dimensions: int,
    label: str,
) -> None:
    raise ValueError(
        f"Expected embedding dimension {expected_dimensions} for {label}, got {actual_dimensions}"
    )


def _raise_embedding_count_error(
    *,
    expected_count: int,
    actual_count: int,
) -> None:
    raise ValueError(f"Expected {expected_count} embeddings, got {actual_count}")


def _extract_response_data(response: EmbeddingResponse) -> list[Any]:
    data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise TypeError("LiteLLM embedding response did not contain a data list")
    return data


def _extract_raw_embedding_from_item(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("embedding")
    return getattr(item, "embedding", None)


def _coerce_float_embedding(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(
            f"LiteLLM embedding response item for {label} did not contain an embedding list"
        )

    embedding: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(
                f"Embedding value at {label}[{index}] must be numeric, got {type(item).__name__}"
            )
        embedding.append(float(item))

    return embedding


def _extract_embedding_from_item(item: Any, *, label: str) -> list[float]:
    return _coerce_float_embedding(
        _extract_raw_embedding_from_item(item),
        label=label,
    )


def _validate_embedding_dimensions(
    embedding: list[float],
    *,
    expected_dimensions: int,
    label: str,
) -> None:
    actual_dimensions = len(embedding)
    if actual_dimensions != expected_dimensions:
        _raise_embedding_dimension_error(
            expected_dimensions=expected_dimensions,
            actual_dimensions=actual_dimensions,
            label=label,
        )


def _validate_embedding_count(
    embeddings: list[list[float]],
    *,
    expected_count: int,
) -> None:
    actual_count = len(embeddings)
    if actual_count != expected_count:
        _raise_embedding_count_error(
            expected_count=expected_count,
            actual_count=actual_count,
        )


class EmbeddingService:
    """Service for generating embeddings using LiteLLM proxy routes."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or settings.embedding

    @observe(as_type="embedding", name="pgvector.generate_embedding")
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding for text."""
        try:
            response = cast(
                "EmbeddingResponse",
                await aembedding(
                    model=self.config.model,
                    input=[text],
                    api_base=self.config.base_url,
                    api_key=self.config.api_key,
                ),
            )
        except Exception as exc:
            raise RuntimeError("Failed to generate embedding") from exc
        else:
            data = _extract_response_data(response)
            if not data:
                raise RuntimeError("LiteLLM embedding response did not contain data")

            embedding = _extract_embedding_from_item(data[0], label="text")
            _validate_embedding_dimensions(
                embedding,
                expected_dimensions=self.config.dimensions,
                label="text",
            )
            _log.debug(
                "Generated embedding: dimensions=%s",
                len(embedding),
            )
            return embedding

    @observe(as_type="embedding", name="pgvector.generate_embeddings_batch")
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        try:
            response = cast(
                "EmbeddingResponse",
                await aembedding(
                    model=self.config.model,
                    input=texts,
                    api_base=self.config.base_url,
                    api_key=self.config.api_key,
                ),
            )
        except Exception as exc:
            raise RuntimeError("Failed to generate embeddings") from exc
        else:
            data = _extract_response_data(response)
            embeddings = [
                _extract_embedding_from_item(item, label=f"text {index}")
                for index, item in enumerate(data)
            ]
            _validate_embedding_count(
                embeddings,
                expected_count=len(texts),
            )

            for index, embedding in enumerate(embeddings):
                _validate_embedding_dimensions(
                    embedding,
                    expected_dimensions=self.config.dimensions,
                    label=f"text {index}",
                )

            _log.debug(
                "Generated embedding batch: count=%s dimensions=%s",
                len(embeddings),
                self.config.dimensions,
            )
            return embeddings

    def update_config(self, new_config: EmbeddingConfig) -> None:
        """Update embedding configuration."""
        self.config = new_config


embedding_service = EmbeddingService()

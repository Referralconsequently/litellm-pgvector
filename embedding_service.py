import logging

import litellm
from config import EmbeddingConfig, settings
from langfuse import observe
from litellm.types.utils import EmbeddingResponse


class EmbeddingService:
    """Service for generating embeddings using OpenAI SDK pointed at LiteLLM proxy"""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or settings.embedding

    @observe(as_type="embedding", name="pgvector.generate_embedding")
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text using LiteLLM proxy

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response: EmbeddingResponse = await litellm.aembedding(
                model=self.config.model,
                input=[text],
                api_base=self.config.base_url,
                api_key=self.config.api_key,
            )
            logging.debug(f"Embedding response: {response}")

            # Extract embedding from response
            embedding = response.data[0]["embedding"]

            # Validate embedding dimensions
            if len(embedding) != self.config.dimensions:
                raise ValueError(
                    f"Expected embedding dimension {self.config.dimensions}, got {len(embedding)}"
                )

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e!s}")

    @observe(as_type="embedding", name="pgvector.generate_embeddings_batch")
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings using LiteLLM
            response = await litellm.aembedding(
                model=self.config.model,
                input=texts,
                api_base=self.config.base_url,
                api_key=self.config.api_key,
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            # Validate embedding dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.config.dimensions:
                    raise ValueError(
                        f"Expected embedding dimension {self.config.dimensions} for text {i}, "
                        f"got {len(embedding)}"
                    )

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e!s}")

    def update_config(self, new_config: EmbeddingConfig) -> None:
        """Update the embedding configuration"""
        self.config = new_config


# Global embedding service instance
embedding_service = EmbeddingService()

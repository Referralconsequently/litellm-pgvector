import os
import time
from datetime import UTC, datetime

# --- Observability bootstrap ---
# Must run before any FastAPI/Prisma import so instrument_httpx() patches the
# httpx client class before Prisma constructs its engine client. The Prisma
# Python client speaks to its Rust query-engine sidecar over local HTTP via
# httpx.AsyncClient, so instrument_httpx covers DB roundtrips. The same
# instrumentor also captures the outbound litellm.aembedding HTTP call to the
# proxy and injects a W3C traceparent header for cross-service stitching.
import logfire


logfire.configure(
    service_name="m2-litellm-pgvector",
    environment=os.getenv("OTEL_ENVIRONMENT_NAME", "local-dev"),
    distributed_tracing=True,
)
logfire.instrument_httpx()
# --- end observability bootstrap ---

from config import settings
from dotenv import load_dotenv
from embedding_service import embedding_service
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse import observe
from models import (
    ContentChunk,
    EmbeddingBatchCreateRequest,
    EmbeddingBatchCreateResponse,
    EmbeddingCreateRequest,
    EmbeddingResponse,
    SearchResult,
    VectorStoreCreateRequest,
    VectorStoreListResponse,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
)
from prisma import Prisma


load_dotenv()

app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0",
)

# Emit one server span per request, excluding the noisy health probe.
logfire.instrument_fastapi(app, excluded_urls="/health")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Prisma client
db = Prisma()

security = HTTPBearer()


def to_epoch_seconds(value):
    """Normalize raw Prisma timestamp values for OpenAI-compatible responses."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp())
    return int(value.timestamp())


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key from Authorization header"""
    expected_key = settings.server_api_key
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@app.on_event("startup")
async def startup() -> None:
    """Connect to database on startup"""
    await db.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    """Disconnect from database on shutdown"""
    await db.disconnect()


async def generate_query_embedding(query: str) -> list[float]:
    """
    Generate an embedding for the query using LiteLLM
    """
    return await embedding_service.generate_embedding(query)


@app.post(
    "/v1/vector_stores",
    response_model=VectorStoreResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_vector_store(request: VectorStoreCreateRequest):
    """
    Create a new vector store.
    """
    try:
        # Use raw SQL to insert the vector store with configurable table/field names
        vector_store_table = settings.table_names["vector_stores"]

        result = await db.query_raw(
            f"""
            INSERT INTO {vector_store_table} (id, name, file_counts, status, usage_bytes, expires_after, metadata, created_at)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())
            RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
                     EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            request.name,
            {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "completed",
            0,
            request.expires_after,
            request.metadata or {},
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create vector store")

        vector_store = result[0]

        # Convert to response format
        created_at = int(vector_store["created_at_timestamp"])
        expires_at = to_epoch_seconds(vector_store.get("expires_at"))
        last_active_at = to_epoch_seconds(vector_store.get("last_active_at"))

        return VectorStoreResponse(
            id=vector_store["id"],
            created_at=created_at,
            name=vector_store["name"],
            usage_bytes=vector_store["usage_bytes"] or 0,
            file_counts=vector_store["file_counts"]
            or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status=vector_store["status"],
            expires_after=vector_store["expires_after"],
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=vector_store["metadata"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e!s}")


@app.get(
    "/v1/vector_stores",
    response_model=VectorStoreListResponse,
    dependencies=[Depends(get_api_key)],
)
async def list_vector_stores(
    limit: int | None = 20,
    after: str | None = None,
    before: str | None = None,
):
    """
    List vector stores with optional pagination.
    """
    try:
        limit = min(limit or 20, 100)  # Cap at 100 results

        vector_store_table = settings.table_names["vector_stores"]

        # Build base query
        base_query = f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        """

        # Add pagination conditions
        conditions = []
        params = []
        param_count = 1

        if after:
            conditions.append(f"id > ${param_count}")
            params.append(after)
            param_count += 1

        if before:
            conditions.append(f"id < ${param_count}")
            params.append(before)
            param_count += 1

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        # Add ordering and limit
        final_query = base_query + f" ORDER BY created_at DESC LIMIT {limit + 1}"

        # Execute query
        results = await db.query_raw(final_query, *params)

        # Check if there are more results
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove extra result

        # Convert to response format
        vector_stores = []
        for row in results:
            created_at = int(row["created_at_timestamp"])
            expires_at = to_epoch_seconds(row.get("expires_at"))
            last_active_at = to_epoch_seconds(row.get("last_active_at"))

            vector_store = VectorStoreResponse(
                id=row["id"],
                created_at=created_at,
                name=row["name"],
                usage_bytes=row["usage_bytes"] or 0,
                file_counts=row["file_counts"]
                or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                status=row["status"],
                expires_after=row["expires_after"],
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=row["metadata"],
            )
            vector_stores.append(vector_store)

        # Determine first_id and last_id
        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None

        return VectorStoreListResponse(
            data=vector_stores, first_id=first_id, last_id=last_id, has_more=has_more
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list vector stores: {e!s}")


@app.post(
    "/v1/vector_stores/{vector_store_id}/search",
    response_model=VectorStoreSearchResponse,
    dependencies=[Depends(get_api_key)],
)
@app.post(
    "/vector_stores/{vector_store_id}/search",
    response_model=VectorStoreSearchResponse,
    dependencies=[Depends(get_api_key)],
)
@observe(as_type="retriever", name="pgvector.search")
async def search_vector_store(vector_store_id: str, request: VectorStoreSearchRequest):
    """
    Search a vector store for similar content.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1", vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Generate embedding for query
        query_embedding = await generate_query_embedding(request.query)
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results

        # Base query with vector similarity using cosine distance
        # Use configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        # Build query with proper parameter placeholders for Prisma
        param_count = 1
        query_params = [query_vector_str, vector_store_id]

        base_query = f"""
        SELECT
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name}
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
        """
        param_count += 2

        # Add metadata filters if provided
        filter_conditions = []

        if request.filters:
            for key, value in request.filters.items():
                filter_conditions.append(
                    f"{fields.metadata_field}->>${param_count} = ${param_count + 1}"
                )
                query_params.extend([key, str(value)])
                param_count += 2

        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)

        # Add ordering and limit
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"

        # Execute the query
        results = await db.query_raw(final_query, *query_params)

        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Convert distance to similarity score (1 - normalized_distance)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            similarity_score = max(0, 1 - (row["distance"] / 2))

            # Extract filename from metadata or use a default
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get("filename", "document.txt")

            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]

            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=similarity_score,
                attributes=metadata if request.return_metadata else None,
                content=content_chunks,
            )
            search_results.append(result)

        return VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,  # TODO: Implement pagination
            next_page=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {e!s}")


@app.post(
    "/v1/vector_stores/{vector_store_id}/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_embedding(vector_store_id: str, request: EmbeddingCreateRequest):
    """
    Add a single embedding to a vector store.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1", vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Convert embedding to vector string format
        embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"

        # Insert embedding using configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {},
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embedding")

        embedding = result[0]

        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            request.content,
        )

        return EmbeddingResponse(
            id=embedding[fields.id_field],
            vector_store_id=embedding[fields.vector_store_id_field],
            content=embedding[fields.content_field],
            metadata=embedding[fields.metadata_field],
            created_at=int(embedding["created_at_timestamp"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {e!s}")


@app.post(
    "/v1/vector_stores/{vector_store_id}/embeddings/batch",
    response_model=EmbeddingBatchCreateResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_embeddings_batch(vector_store_id: str, request: EmbeddingBatchCreateRequest):
    """
    Add multiple embeddings to a vector store in batch.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1", vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if not request.embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided")

        # Prepare batch insert
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        # Build VALUES clause for batch insert
        values_clauses = []
        params = []
        param_count = 1

        for embedding_req in request.embeddings:
            embedding_vector_str = "[" + ",".join(map(str, embedding_req.embedding)) + "]"
            values_clauses.append(
                f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}::vector, ${param_count + 3}, NOW())"
            )
            params.extend(
                [
                    vector_store_id,
                    embedding_req.content,
                    embedding_vector_str,
                    embedding_req.metadata or {},
                ]
            )
            param_count += 4

        values_clause = ", ".join(values_clauses)

        # Execute batch insert
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES {values_clause}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            *params,
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")

        # Calculate total content length for usage bytes update
        total_content_length = sum(len(emb.content) for emb in request.embeddings)

        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + $3,
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            len(request.embeddings),
            total_content_length,
        )

        # Convert results to response format
        embeddings = []
        for row in result:
            embeddings.append(
                EmbeddingResponse(
                    id=row[fields.id_field],
                    vector_store_id=row[fields.vector_store_id_field],
                    content=row[fields.content_field],
                    metadata=row[fields.metadata_field],
                    created_at=int(row["created_at_timestamp"]),
                )
            )

        return EmbeddingBatchCreateResponse(data=embeddings, created=int(time.time()))

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings batch: {e!s}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)

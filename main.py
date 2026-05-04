from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, NoReturn, cast

import langfuse
import logfire
import uvicorn
from config import settings
from dotenv import load_dotenv
from embedding_service import embedding_service
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


Row = dict[str, Any]
_observe = cast("Any", langfuse.observe)

_log = logging.getLogger(__name__)
_SQL_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_FILE_COUNTS: Final = {
    "in_progress": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "total": 0,
}


@dataclass(frozen=True)
class SafeEmbeddingFields:
    id_field: str
    vector_store_id_field: str
    content_field: str
    embedding_field: str
    metadata_field: str
    created_at_field: str


def configure_observability() -> None:
    """Configure Logfire before Prisma/FastAPI clients are constructed."""
    logfire.configure(
        service_name="m2-litellm-pgvector",
        environment=os.getenv("OTEL_ENVIRONMENT_NAME", "local-dev"),
        distributed_tracing=True,
        inspect_arguments=False,
    )
    logfire.instrument_httpx()


def sql_identifier(value: str, *, label: str) -> str:
    """Validate a SQL identifier before interpolating it into raw SQL.

    Query parameters can bind values, but not table or column identifiers.
    Every table or column name interpolated into query_raw must pass here first.
    """
    if not _SQL_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier for {label}: {value!r}")
    return value


def sql_template(template: str, **identifiers: str) -> str:
    """Substitute validated SQL identifiers into a static SQL template."""
    sql = dedent(template).strip()
    for key, value in identifiers.items():
        sql = sql.replace(
            "{{" + key + "}}",
            sql_identifier(value, label=key),
        )
    return sql


def raise_http_exception(*, status_code: int, detail: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail=detail)


def default_file_counts() -> dict[str, int]:
    return dict(_DEFAULT_FILE_COUNTS)


def row_required(
    rows: list[Row],
    *,
    status_code: int,
    detail: str,
) -> Row:
    if not rows:
        raise_http_exception(status_code=status_code, detail=detail)
    return rows[0]


def row_metadata(value: Any) -> Row:
    if isinstance(value, dict):
        return cast("Row", value)
    return {}


def vector_store_table_name() -> str:
    return sql_identifier(
        settings.table_names["vector_stores"],
        label="vector_stores table",
    )


def embeddings_table_name() -> str:
    return sql_identifier(
        settings.table_names["embeddings"],
        label="embeddings table",
    )


def safe_embedding_fields() -> SafeEmbeddingFields:
    fields = settings.db_fields
    return SafeEmbeddingFields(
        id_field=sql_identifier(fields.id_field, label="embedding id field"),
        vector_store_id_field=sql_identifier(
            fields.vector_store_id_field,
            label="embedding vector_store_id field",
        ),
        content_field=sql_identifier(fields.content_field, label="embedding content field"),
        embedding_field=sql_identifier(
            fields.embedding_field,
            label="embedding vector field",
        ),
        metadata_field=sql_identifier(
            fields.metadata_field,
            label="embedding metadata field",
        ),
        created_at_field=sql_identifier(
            fields.created_at_field,
            label="embedding created_at field",
        ),
    )


def json_metadata_filter_condition(
    metadata_field: str,
    *,
    key_param: int,
    value_param: int,
) -> str:
    return metadata_field + "->>$" + str(key_param) + " = $" + str(value_param)


def batch_values_clause(param_count: int) -> str:
    return (
        "(gen_random_uuid(), $"
        + str(param_count)
        + ", $"
        + str(param_count + 1)
        + ", $"
        + str(param_count + 2)
        + "::vector, $"
        + str(param_count + 3)
        + ", NOW())"
    )


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def to_epoch_seconds(value: Any) -> int | None:
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
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(parsed.timestamp())

    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        timestamp_value = timestamp()
        if isinstance(timestamp_value, (int, float, str)):
            return int(timestamp_value)

    raise TypeError(f"Cannot convert {type(value).__name__} to epoch seconds")


def vector_store_response_from_row(row: Row) -> VectorStoreResponse:
    return VectorStoreResponse(
        id=row["id"],
        created_at=int(row["created_at_timestamp"]),
        name=row["name"],
        usage_bytes=row["usage_bytes"] or 0,
        file_counts=row["file_counts"] or default_file_counts(),
        status=row["status"],
        expires_after=row["expires_after"],
        expires_at=to_epoch_seconds(row.get("expires_at")),
        last_active_at=to_epoch_seconds(row.get("last_active_at")),
        metadata=row["metadata"],
    )


def search_result_from_row(
    row: Row,
    *,
    fields: SafeEmbeddingFields,
    return_metadata: bool,
) -> SearchResult:
    distance = float(row["distance"])
    similarity_score = max(0.0, 1.0 - (distance / 2.0))
    metadata = row_metadata(row[fields.metadata_field])
    filename = metadata.get("filename")
    if not isinstance(filename, str) or not filename:
        filename = "document.txt"

    return SearchResult(
        file_id=row[fields.id_field],
        filename=filename,
        score=similarity_score,
        attributes=metadata if return_metadata else None,
        content=[ContentChunk(type="text", text=row[fields.content_field])],
    )


def embedding_response_from_row(row: Row, *, fields: SafeEmbeddingFields) -> EmbeddingResponse:
    return EmbeddingResponse(
        id=row[fields.id_field],
        vector_store_id=row[fields.vector_store_id_field],
        content=row[fields.content_field],
        metadata=row[fields.metadata_field],
        created_at=int(row["created_at_timestamp"]),
    )


load_dotenv()
configure_observability()

db = Prisma()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Connect and disconnect Prisma with the FastAPI application lifecycle."""
    await db.connect()
    try:
        yield
    finally:
        await db.disconnect()


app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0",
    lifespan=lifespan,
)

# Emit one server span per request, excluding the noisy health probe.
logfire.instrument_fastapi(app, excluded_urls=r".*/health$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate API key from Authorization header."""
    expected_key = settings.server_api_key
    if credentials.credentials != expected_key:
        raise_http_exception(status_code=401, detail="Invalid API key")
    return credentials.credentials


async def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding for the query using LiteLLM."""
    return await embedding_service.generate_embedding(query)


@app.post(
    "/v1/vector_stores",
    response_model=VectorStoreResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_vector_store(
    request: VectorStoreCreateRequest,
) -> VectorStoreResponse:
    """Create a new vector store."""
    try:
        vector_store_table = vector_store_table_name()
        query = sql_template(
            """
            INSERT INTO {{vector_store_table}} (
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                metadata,
                created_at
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())
            RETURNING
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                EXTRACT(EPOCH FROM created_at)::bigint AS created_at_timestamp
            """,
            vector_store_table=vector_store_table,
        )
        result = await db.query_raw(
            query,
            request.name,
            default_file_counts(),
            "completed",
            0,
            request.expires_after,
            request.metadata or {},
        )
        vector_store = row_required(
            result,
            status_code=500,
            detail="Failed to create vector store",
        )
        response = vector_store_response_from_row(vector_store)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to create vector store")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vector store: {exc!s}",
        ) from exc
    else:
        return response


@app.get(
    "/v1/vector_stores",
    response_model=VectorStoreListResponse,
    dependencies=[Depends(get_api_key)],
)
async def list_vector_stores(
    limit: int | None = 20,
    after: str | None = None,
    before: str | None = None,
) -> VectorStoreListResponse:
    """List vector stores with optional pagination."""
    try:
        page_size = min(limit or 20, 100)
        vector_store_table = vector_store_table_name()

        base_query = sql_template(
            """
            SELECT
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                EXTRACT(EPOCH FROM created_at)::bigint AS created_at_timestamp
            FROM {{vector_store_table}}
            """,
            vector_store_table=vector_store_table,
        )

        conditions: list[str] = []
        params: list[Any] = []
        param_count = 1

        if after:
            conditions.append("id > $" + str(param_count))
            params.append(after)
            param_count += 1

        if before:
            conditions.append("id < $" + str(param_count))
            params.append(before)
            param_count += 1

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        params.append(page_size + 1)
        final_query = base_query + " ORDER BY created_at DESC LIMIT $" + str(param_count)

        results = await db.query_raw(final_query, *params)
        has_more = len(results) > page_size
        if has_more:
            results = results[:page_size]

        vector_stores = [vector_store_response_from_row(row) for row in results]
        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None
        response = VectorStoreListResponse(
            data=vector_stores,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to list vector stores")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list vector stores: {exc!s}",
        ) from exc
    else:
        return response


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
@_observe(as_type="retriever", name="pgvector.search")
async def search_vector_store(
    vector_store_id: str,
    request: VectorStoreSearchRequest,
) -> VectorStoreSearchResponse:
    """Search a vector store for similar content."""
    try:
        vector_store_table = vector_store_table_name()
        vector_store_query = sql_template(
            "SELECT id FROM {{vector_store_table}} WHERE id = $1",
            vector_store_table=vector_store_table,
        )
        vector_store_result = await db.query_raw(vector_store_query, vector_store_id)
        if not vector_store_result:
            raise_http_exception(status_code=404, detail="Vector store not found")

        query_embedding = await generate_query_embedding(request.query)
        query_vector_str = vector_literal(query_embedding)
        page_size = min(request.limit or 20, 100)

        fields = safe_embedding_fields()
        table_name = embeddings_table_name()
        param_count = 1
        query_params: list[Any] = [query_vector_str, vector_store_id]

        base_query = sql_template(
            """
            SELECT
                {{id_field}},
                {{content_field}},
                {{metadata_field}},
                ({{embedding_field}} <=> $1::vector) AS distance
            FROM {{table_name}}
            WHERE {{vector_store_id_field}} = $2
            """,
            id_field=fields.id_field,
            content_field=fields.content_field,
            metadata_field=fields.metadata_field,
            embedding_field=fields.embedding_field,
            table_name=table_name,
            vector_store_id_field=fields.vector_store_id_field,
        )
        param_count += 2

        filter_conditions: list[str] = []
        if request.filters:
            for key, value in request.filters.items():
                filter_conditions.append(
                    json_metadata_filter_condition(
                        fields.metadata_field,
                        key_param=param_count,
                        value_param=param_count + 1,
                    )
                )
                query_params.extend([key, str(value)])
                param_count += 2

        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)

        query_params.append(page_size)
        final_query = base_query + " ORDER BY distance ASC LIMIT $" + str(param_count)
        results = await db.query_raw(final_query, *query_params)

        search_results = [
            search_result_from_row(
                row,
                fields=fields,
                return_metadata=request.return_metadata or False,
            )
            for row in results
        ]

        response = VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,
            next_page=None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc!s}") from exc
    else:
        return response


@app.post(
    "/v1/vector_stores/{vector_store_id}/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_embedding(
    vector_store_id: str,
    request: EmbeddingCreateRequest,
) -> EmbeddingResponse:
    """Add a single embedding to a vector store."""
    try:
        vector_store_table = vector_store_table_name()
        vector_store_query = sql_template(
            "SELECT id FROM {{vector_store_table}} WHERE id = $1",
            vector_store_table=vector_store_table,
        )
        vector_store_result = await db.query_raw(vector_store_query, vector_store_id)
        if not vector_store_result:
            raise_http_exception(status_code=404, detail="Vector store not found")

        embedding_vector_str = vector_literal(request.embedding)
        fields = safe_embedding_fields()
        table_name = embeddings_table_name()

        insert_query = sql_template(
            """
            INSERT INTO {{table_name}} (
                {{id_field}},
                {{vector_store_id_field}},
                {{content_field}},
                {{embedding_field}},
                {{metadata_field}},
                {{created_at_field}}
            )
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
            RETURNING
                {{id_field}},
                {{vector_store_id_field}},
                {{content_field}},
                {{metadata_field}},
                EXTRACT(EPOCH FROM {{created_at_field}})::bigint
                    AS created_at_timestamp
            """,
            table_name=table_name,
            id_field=fields.id_field,
            vector_store_id_field=fields.vector_store_id_field,
            content_field=fields.content_field,
            embedding_field=fields.embedding_field,
            metadata_field=fields.metadata_field,
            created_at_field=fields.created_at_field,
        )
        result = await db.query_raw(
            insert_query,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {},
        )
        embedding = row_required(
            result,
            status_code=500,
            detail="Failed to create embedding",
        )

        update_query = sql_template(
            """
            UPDATE {{vector_store_table}}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(
                            file_counts,
                            '{"in_progress": 0, "completed": 0, "failed": 0,
                              "cancelled": 0, "total": 0}'::jsonb
                        ),
                        '{completed}',
                        (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                    ),
                    '{total}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_table=vector_store_table,
        )
        await db.query_raw(update_query, vector_store_id, request.content)

        response = embedding_response_from_row(embedding, fields=fields)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to create embedding")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create embedding: {exc!s}",
        ) from exc
    else:
        return response


@app.post(
    "/v1/vector_stores/{vector_store_id}/embeddings/batch",
    response_model=EmbeddingBatchCreateResponse,
    dependencies=[Depends(get_api_key)],
)
async def create_embeddings_batch(
    vector_store_id: str,
    request: EmbeddingBatchCreateRequest,
) -> EmbeddingBatchCreateResponse:
    """Add multiple embeddings to a vector store in batch."""
    try:
        vector_store_table = vector_store_table_name()
        vector_store_query = sql_template(
            "SELECT id FROM {{vector_store_table}} WHERE id = $1",
            vector_store_table=vector_store_table,
        )
        vector_store_result = await db.query_raw(vector_store_query, vector_store_id)
        if not vector_store_result:
            raise_http_exception(status_code=404, detail="Vector store not found")

        if not request.embeddings:
            raise_http_exception(status_code=400, detail="No embeddings provided")

        fields = safe_embedding_fields()
        table_name = embeddings_table_name()
        values_clauses: list[str] = []
        params: list[Any] = []
        param_count = 1

        for embedding_req in request.embeddings:
            embedding_vector_str = vector_literal(embedding_req.embedding)
            values_clauses.append(batch_values_clause(param_count))
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
        insert_query = sql_template(
            """
            INSERT INTO {{table_name}} (
                {{id_field}},
                {{vector_store_id_field}},
                {{content_field}},
                {{embedding_field}},
                {{metadata_field}},
                {{created_at_field}}
            )
            VALUES {{values_clause}}
            RETURNING
                {{id_field}},
                {{vector_store_id_field}},
                {{content_field}},
                {{metadata_field}},
                EXTRACT(EPOCH FROM {{created_at_field}})::bigint
                    AS created_at_timestamp
            """,
            table_name=table_name,
            id_field=fields.id_field,
            vector_store_id_field=fields.vector_store_id_field,
            content_field=fields.content_field,
            embedding_field=fields.embedding_field,
            metadata_field=fields.metadata_field,
            created_at_field=fields.created_at_field,
        ).replace("{{values_clause}}", values_clause)

        result = await db.query_raw(insert_query, *params)
        if not result:
            raise_http_exception(status_code=500, detail="Failed to create embeddings")

        total_content_length = sum(len(embedding.content) for embedding in request.embeddings)
        update_query = sql_template(
            """
            UPDATE {{vector_store_table}}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(
                            file_counts,
                            '{"in_progress": 0, "completed": 0, "failed": 0,
                              "cancelled": 0, "total": 0}'::jsonb
                        ),
                        '{completed}',
                        (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                    ),
                    '{total}',
                    (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + $3,
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_table=vector_store_table,
        )
        await db.query_raw(
            update_query,
            vector_store_id,
            len(request.embeddings),
            total_content_length,
        )

        embeddings = [embedding_response_from_row(row, fields=fields) for row in result]
        response = EmbeddingBatchCreateResponse(
            data=embeddings,
            created=int(time.time()),
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to create embeddings batch")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create embeddings batch: {exc!s}",
        ) from exc
    else:
        return response


@app.get("/health")
async def health_check() -> dict[str, int | str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)

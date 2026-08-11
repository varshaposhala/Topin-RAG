"""Pinecone adapter with a Qdrant-like interface used by app.py.

Uses ONE namespace (free-tier friendly). Topic/collection is stored in metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from pinecone import Pinecone, ServerlessSpec

VECTOR_SIZE = 384
DEFAULT_NAMESPACE = "questions"
MAX_PAGE_CONTENT_CHARS = 35000


@dataclass
class PineconePoint:
    id: str
    payload: dict
    score: float = 0.0


@dataclass
class QueryResult:
    points: list[PineconePoint] = field(default_factory=list)


def _flatten_payload(page_content: str, metadata: dict | None, collection: str | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    if collection:
        meta["collection"] = collection
    flat: dict[str, Any] = {
        "page_content": (page_content or "")[:MAX_PAGE_CONTENT_CHARS],
    }
    for key in (
        "collection",
        "question_id",
        "topic",
        "subtopic",
        "difficulty",
        "unit_tag",
        "module_tag",
        "course_tag",
        "grit_tag",
        "extra_tags",
    ):
        value = meta.get(key, "")
        if value is None:
            continue
        text = str(value)
        if text and text.lower() not in {"nan", "none"}:
            flat[key] = text
    return flat


def _payload_from_metadata(meta: dict | None) -> dict:
    meta = meta or {}
    page_content = meta.get("page_content", "") or ""
    metadata = {
        key: value
        for key, value in meta.items()
        if key != "page_content" and value is not None and str(value).lower() not in {"nan", "none", ""}
    }
    return {"page_content": page_content, "metadata": metadata}


class PineconeVectorDB:
    """Drop-in replacement for the Qdrant client methods used by the app."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        recreate: bool = False,
    ):
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing. Add it to .streamlit/secrets.toml")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME is missing. Add it to .streamlit/secrets.toml")

        self.api_key = api_key
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.namespace = DEFAULT_NAMESPACE
        self.pc = Pinecone(api_key=api_key)
        if recreate and self.pc.has_index(index_name):
            self.pc.delete_index(index_name)
            time.sleep(5)
        self._ensure_index()
        self.index = self.pc.Index(index_name)
        self._collection_counts: dict[str, int] = {}

    def _ensure_index(self) -> None:
        if self.pc.has_index(self.index_name):
            return
        self.pc.create_index(
            name=self.index_name,
            dimension=VECTOR_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        for _ in range(60):
            desc = self.pc.describe_index(self.index_name)
            status = getattr(desc, "status", None)
            ready = bool(status.get("ready")) if isinstance(status, dict) else bool(getattr(status, "ready", False))
            if ready:
                break
            time.sleep(2)

    def set_collection_counts(self, counts: dict[str, int]) -> None:
        self._collection_counts = dict(counts)

    def get_collections(self):
        names = sorted(self._collection_counts.keys()) or ["all_questions"]
        if "all_questions" not in names:
            names.append("all_questions")
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in names])

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def create_collection(self, collection_name: str, vectors_config=None) -> None:
        return None

    def get_collection(self, name: str):
        if name == "all_questions":
            total = sum(v for k, v in self._collection_counts.items() if k != "all_questions")
            if total:
                return SimpleNamespace(points_count=total)
            stats = self.index.describe_index_stats()
            return SimpleNamespace(points_count=int(getattr(stats, "total_vector_count", 0) or 0))
        return SimpleNamespace(points_count=int(self._collection_counts.get(name, 0)))

    def query_points(
        self,
        collection_name: str,
        query,
        limit: int = 10,
        with_payload: bool = True,
        **kwargs,
    ) -> QueryResult:
        query_kwargs: dict[str, Any] = {
            "vector": list(query),
            "top_k": max(1, int(limit)),
            "namespace": self.namespace,
            "include_metadata": bool(with_payload),
        }
        if collection_name and collection_name != "all_questions":
            query_kwargs["filter"] = {"collection": {"$eq": collection_name}}

        response = self.index.query(**query_kwargs)
        matches = getattr(response, "matches", None) or response.get("matches", [])
        points: list[PineconePoint] = []
        for match in matches:
            if isinstance(match, dict):
                mid = match.get("id")
                score = float(match.get("score") or 0.0)
                meta = match.get("metadata") or {}
            else:
                mid = getattr(match, "id", None)
                score = float(getattr(match, "score", 0.0) or 0.0)
                meta = getattr(match, "metadata", None) or {}
            points.append(
                PineconePoint(
                    id=str(mid),
                    score=score,
                    payload=_payload_from_metadata(meta),
                )
            )
        return QueryResult(points=points)

    def fetch_ids(self, collection_name: str, ids: list[str]) -> list[PineconePoint]:
        if not ids:
            return []
        points: list[PineconePoint] = []
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            response = self.index.fetch(ids=batch, namespace=self.namespace)
            vectors = getattr(response, "vectors", None) or response.get("vectors", {}) or {}
            for vid, data in vectors.items():
                meta = data.get("metadata") if isinstance(data, dict) else (getattr(data, "metadata", None) or {})
                payload = _payload_from_metadata(meta)
                stored_collection = (payload.get("metadata") or {}).get("collection", "")
                if (
                    collection_name
                    and collection_name != "all_questions"
                    and stored_collection
                    and stored_collection != collection_name
                ):
                    continue
                points.append(PineconePoint(id=str(vid), payload=payload, score=1.0))
        return points

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ):
        """
        Paginate through the single namespace, optionally filtering by metadata.collection.
        `offset` is a dict: {"token": str|None, "buffer": list[PineconePoint]}
        """
        state = offset if isinstance(offset, dict) else {"token": offset, "buffer": []}
        buffer: list[PineconePoint] = list(state.get("buffer") or [])
        token = state.get("token")

        while len(buffer) < limit:
            list_kwargs: dict[str, Any] = {
                "namespace": self.namespace,
                "limit": 100,
            }
            if token:
                list_kwargs["pagination_token"] = token

            try:
                page = self.index.list_paginated(**list_kwargs)
            except Exception:
                ids: list[str] = []
                for batch in self.index.list(namespace=self.namespace, limit=100):
                    ids.extend(batch)
                    break
                fetched = self.fetch_ids("all_questions", ids) if with_payload else [
                    PineconePoint(id=i, payload={}, score=1.0) for i in ids
                ]
                if collection_name and collection_name != "all_questions":
                    fetched = [
                        p
                        for p in fetched
                        if (p.payload.get("metadata") or {}).get("collection") == collection_name
                    ]
                return fetched[:limit], None

            vectors = getattr(page, "vectors", None) or getattr(page, "data", None) or []
            ids = []
            for item in vectors:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    ids.append(str(item.get("id")))
                else:
                    ids.append(str(getattr(item, "id", item)))

            pagination = getattr(page, "pagination", None)
            next_token = None
            if pagination is not None:
                next_token = getattr(pagination, "next", None)
                if next_token is None and isinstance(pagination, dict):
                    next_token = pagination.get("next")

            if not ids:
                break

            fetched = self.fetch_ids("all_questions", ids) if with_payload else [
                PineconePoint(id=i, payload={}, score=1.0) for i in ids
            ]
            if collection_name and collection_name != "all_questions":
                fetched = [
                    p
                    for p in fetched
                    if (p.payload.get("metadata") or {}).get("collection") == collection_name
                ]
            buffer.extend(fetched)
            token = next_token
            if not next_token:
                break

        page_points = buffer[:limit]
        rest = buffer[limit:]
        next_offset = None
        if rest or token:
            next_offset = {"token": token, "buffer": rest}
        return page_points, next_offset

    def upsert_records(self, collection_name: str, records: list[dict]) -> None:
        vectors = []
        for record in records:
            vectors.append(
                {
                    "id": str(record["id"]),
                    "values": list(record["values"]),
                    "metadata": _flatten_payload(
                        record.get("page_content", ""),
                        record.get("metadata"),
                        collection=collection_name,
                    ),
                }
            )
        for start in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[start : start + 100], namespace=self.namespace)


class SimpleVectorStore:
    """Tiny replacement for LangChain vector store similarity_search."""

    def __init__(self, client: PineconeVectorDB, embeddings, collection_name: str = "all_questions"):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name

    def similarity_search(self, query: str, k: int = 10):
        vector = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=k,
            with_payload=True,
        )
        return [
            SimpleNamespace(
                page_content=point.payload.get("page_content", ""),
                metadata=point.payload.get("metadata", {}) or {},
            )
            for point in results.points
        ]

"""FastAPI backend for Topin Question Engine (same logic as Streamlit app)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TOPIN_API_MODE"] = "1"

from backend.config import apply_secrets_to_environ  # noqa: E402
from backend.hf_embeddings import create_embeddings  # noqa: E402
from backend.streamlit_stub import install_streamlit_stub  # noqa: E402

_secrets = apply_secrets_to_environ()
install_streamlit_stub(_secrets)

import app as engine  # noqa: E402

app = FastAPI(title="Topin Question Engine API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None
_embeddings = None
_SESSIONS: dict[str, dict] = {}


def get_client():
    global _client
    if _client is None:
        _client = engine.make_db_client()
    return _client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = create_embeddings()
    return _embeddings


class _LazyEmbeddings:
    """Load the embedding model only when a vector call is actually needed."""

    def embed_query(self, text: str) -> list[float]:
        return get_embeddings().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings().embed_documents(texts)


@app.get("/api/health")
def health():
    """Lightweight health check for Render / load balancers."""
    return {
        "ok": True,
        "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "fast"),
    }


@app.on_event("startup")
def warm_up() -> None:
    """Warm DB + CSV indexes; embeddings load lazily on first vector search."""
    try:
        get_client()
        engine.load_topic_catalog()
        engine.load_question_tag_index()
    except Exception as exc:  # noqa: BLE001
        print(f"[warm_up] non-fatal startup warm-up failure: {exc}", flush=True)


class SelectionPayload(BaseModel):
    topic: str | None = None
    subject: str | None = None
    question_type: str | None = None
    count_choice: str | None = None
    difficulty: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    selections: SelectionPayload | None = None
    partial_intent: dict[str, Any] | None = None
    session_id: str | None = None


def _option_rows(options_str: str) -> list[dict]:
    rows = []
    for text, is_correct in engine.format_options(options_str):
        label = text.split("**")[1].replace(".", "") if "**" in text else ""
        body = text.split(".** ", 1)[-1] if ".** " in text else text
        rows.append({"label": label, "text": body, "is_correct": is_correct})
    return rows


def serialize_results(results: list[dict], matched_tags: list[str] | None = None) -> list[dict]:
    tag_index, tag_display, _, _ = engine.load_question_tag_index()
    serialized = []
    for idx, item in enumerate(results, start=1):
        parsed = engine.parse_question_content(item["content"], item.get("metadata"))
        raw_topic = engine._raw_value(
            parsed.get("topic") or item.get("collection", "").removesuffix("_questions")
        )
        raw_subtopic = engine._raw_value(parsed.get("subtopic", ""))
        difficulty = engine._clean_label(
            parsed.get("difficulty", item.get("metadata", {}).get("difficulty", ""))
        )
        question_text = parsed.get("question_text") or item["content"]
        qid = engine._raw_value(
            parsed.get("question_id") or item.get("metadata", {}).get("question_id", "")
        )
        tags = sorted(tag_index.get(engine.normalize_question_id(qid), set())) if qid else []
        serialized.append(
            {
                "index": idx,
                "question_id": qid,
                "topic": raw_topic,
                "subtopic": raw_subtopic,
                "difficulty": difficulty,
                "question_text": engine.normalize_markdown_question(question_text),
                "short_description": engine._clean_label(parsed.get("short_description", "")),
                "options": _option_rows(parsed.get("options", "")),
                "is_coding": engine.is_coding_question(item),
                "tags": tags,
                "matched_tags": matched_tags or [],
                "collection": item.get("collection", ""),
                "score": float(item.get("score") or 0),
                "all_tags_text": engine.get_tags_text(qid, tag_display) if qid else "",
            }
        )
    return serialized


def store_session(intent: dict, results: list[dict], pool: list[dict], query: str) -> str:
    # Cap memory on small hosts (Render).
    while len(_SESSIONS) >= 8:
        _SESSIONS.pop(next(iter(_SESSIONS)))
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "intent": dict(intent),
        "results": results,
        "result_pool": pool[:500] if pool else pool,
        "shown_ids": {
            engine.get_question_id(item) for item in results if engine.get_question_id(item)
        },
        "query": query,
    }
    return session_id


def catalogs() -> dict:
    topics = engine.load_topic_catalog()
    return {
        "subjects": [{"value": v, "label": l} for v, l in engine.SUBJECT_OPTIONS],
        "topics": [
            {
                "value": topic,
                "label": topic.replace("topic_", "").replace("_", " ").title(),
            }
            for topic in topics
        ],
        "question_types": [{"value": v, "label": l} for v, l in engine.QUESTION_TYPE_OPTIONS],
        "counts": [{"value": v, "label": l} for v, l in engine.COUNT_OPTIONS],
        "difficulties": [{"value": v, "label": l} for v, l in engine.DIFFICULTY_OPTIONS],
    }


@app.get("/api/catalogs")
def get_catalogs():
    return catalogs()


@app.post("/api/search")
def search(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    client = get_client()
    embeddings = _LazyEmbeddings()

    if payload.session_id and payload.session_id in _SESSIONS:
        ctx = _SESSIONS[payload.session_id]
        if engine.parse_follow_up(query, has_context=True):
            follow = engine.handle_follow_up(query, ctx)
            if follow:
                results, label = follow
                intent = ctx["intent"]
                pool = ctx.get("result_pool") or results
                session_id = store_session(intent, results, pool, query)
                _, tag_display, _, _ = engine.load_question_tag_index()
                return {
                    "type": "results",
                    "label": label,
                    "questions": serialize_results(results, intent.get("tags")),
                    "intent": intent,
                    "session_id": session_id,
                    "csv": engine.results_to_csv(results, tag_display) if len(results) <= 200 else "",
                }

    intent = payload.partial_intent or engine.parse_query_intent(query)
    if payload.selections:
        intent = engine.build_intent_from_selection(
            intent, payload.selections.model_dump(exclude_none=True)
        )
        intent = engine.finalize_query_intent(intent, query)

    needs, message = engine.requires_subject_type_selection(query, intent)
    if needs and not payload.selections:
        missing = engine.get_missing_selection_fields(intent, query)
        return {
            "type": "needs_selection",
            "message": message,
            "missing": missing,
            "partial_intent": intent,
            "catalogs": catalogs(),
            "detected": {
                "subject": intent.get("subject"),
                "question_type": intent.get("question_type"),
                "mixed": intent.get("mixed"),
                "tags": intent.get("tags") or [],
                "topics": intent.get("topics") or [],
                "difficulty": intent.get("difficulty"),
                "limit": intent.get("limit"),
                "fetch_all": intent.get("fetch_all"),
            },
        }

    try:
        results, label = engine.search_all_collections(
            client, embeddings, query, intent_override=intent
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    if not results:
        return {
            "type": "empty",
            "message": label or "No matching questions found.",
            "intent": intent,
            "questions": [],
        }

    # Tag / fetch-all searches already return the full set — skip a second full scan.
    if engine.is_tag_primary_intent(intent) or intent.get("fetch_all"):
        pool = results
    elif engine.intent_has_filters(intent):
        try:
            pool = engine.fetch_pool_for_intent(client, embeddings, intent, query)
        except Exception as exc:  # noqa: BLE001
            print(f"[search] pool fetch failed, using results: {exc}", flush=True)
            pool = results
    else:
        pool = results

    # LLM intro is optional; skip when disabled or when it would add latency.
    intro = label
    if os.getenv("SKIP_LLM_INTRO", "").strip() not in {"1", "true", "yes"}:
        try:
            intro = engine.generate_search_intro_llm(
                query,
                intent,
                len(results),
                sorted({item.get("collection", "") for item in results}),
            ) or label
        except Exception:  # noqa: BLE001
            intro = label

    session_id = store_session(intent, results, pool, query)
    _, tag_display, _, _ = engine.load_question_tag_index()
    csv_text = ""
    if len(results) <= 200:
        try:
            csv_text = engine.results_to_csv(results, tag_display)
        except Exception as exc:  # noqa: BLE001
            print(f"[search] csv build failed: {exc}", flush=True)
    return {
        "type": "results",
        "label": intro or label,
        "questions": serialize_results(results, intent.get("tags")),
        "intent": intent,
        "session_id": session_id,
        "csv": csv_text,
    }


@app.post("/api/export")
def export_csv(payload: SearchRequest):
    result = search(payload)
    if result.get("type") != "results":
        raise HTTPException(status_code=400, detail="No results to export")
    return Response(
        content=result["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=topin_questions.csv"},
    )


STATIC_DIR = ROOT / "frontend" / "dist"
if STATIC_DIR.exists():
    assets = STATIC_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def root():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index = STATIC_DIR / "index.html"
        return HTMLResponse(index.read_text(encoding="utf-8"))

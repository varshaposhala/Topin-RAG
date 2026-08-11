"""
Upload Topin questions into Pinecone.

Usage:
  1. Add PINECONE_API_KEY to .streamlit/secrets.toml
  2. python -u reindex_pinecone.py

Optional:
  python -u reindex_pinecone.py --limit 200
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
import urllib.request
import uuid
from collections import defaultdict

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone_db import PineconeVectorDB, VECTOR_SIZE

CSV_PATH = "topin_cleaned_data.csv"
BATCH_SIZE = 64


def log(message: str) -> None:
    print(message, flush=True)


def load_secrets() -> dict:
    with open(".streamlit/secrets.toml", "rb") as handle:
        return tomllib.load(handle)


def ensure_csv(data_link: str | None) -> None:
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        log(f"Using existing {CSV_PATH}")
        return
    if not data_link:
        raise FileNotFoundError(f"{CSV_PATH} missing and no data_link in secrets.")
    log("Downloading CSV from data_link...")
    urllib.request.urlretrieve(data_link, CSV_PATH)
    log(f"Saved {CSV_PATH}")


def raw(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def normalize_question_id(qid: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(qid).lower())


def point_uuid(qid: str) -> str:
    cleaned = normalize_question_id(qid)
    if len(cleaned) == 32:
        return str(uuid.UUID(cleaned))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, cleaned or "empty"))


def collection_for_row(topic: str, options: str) -> str:
    if topic:
        return f"{topic}_questions"
    has_options = bool(options and options.lower() not in {"nan", "none", "[]", "{}"})
    return "unassigned_mcq_questions" if has_options else "unassigned_coding_questions"


def build_page_content(topic, subtopic, difficulty, content, options) -> str:
    lines = []
    if topic:
        lines.append(f"Topic: {topic}")
    if subtopic:
        lines.append(f"Subtopic: {subtopic}")
    if difficulty:
        lines.append(f"Difficulty: {difficulty}")
    if content:
        lines.append(f"Question Text:\n{content}")
    if options:
        lines.append(f"Options: {options}")
    return "\n".join(lines)


def list_existing_vector_ids(db: PineconeVectorDB) -> set[str]:
    """List all vector IDs already stored in the shared namespace."""
    existing: set[str] = set()
    token = None
    while True:
        kwargs = {"namespace": db.namespace, "limit": 100}
        if token:
            kwargs["pagination_token"] = token
        try:
            page = db.index.list_paginated(**kwargs)
        except Exception:
            for batch in db.index.list(namespace=db.namespace, limit=100):
                existing.update(str(item) for item in batch)
            break

        vectors = getattr(page, "vectors", None) or getattr(page, "data", None) or []
        for item in vectors:
            if isinstance(item, str):
                existing.add(item)
            elif isinstance(item, dict):
                existing.add(str(item.get("id")))
            else:
                existing.add(str(getattr(item, "id", item)))

        pagination = getattr(page, "pagination", None)
        token = None
        if pagination is not None:
            token = getattr(pagination, "next", None)
            if token is None and isinstance(pagination, dict):
                token = pagination.get("next")
        if not token:
            break
        if len(existing) % 5000 == 0:
            log(f"  listed {len(existing)} existing IDs...")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Pinecone index (needed after namespace-limit errors)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip vectors already present in Pinecone and upload only remaining ones",
    )
    args = parser.parse_args()

    secrets = load_secrets()
    api_key = secrets.get("PINECONE_API_KEY")
    index_name = secrets.get("PINECONE_INDEX_NAME") or "topin-questions"
    cloud = secrets.get("PINECONE_CLOUD") or "aws"
    region = secrets.get("PINECONE_REGION") or "us-east-1"
    if not api_key or str(api_key).startswith("YOUR_"):
        raise ValueError("Set a real PINECONE_API_KEY in .streamlit/secrets.toml")

    ensure_csv(secrets.get("data_link"))

    log(f"Connecting to Pinecone index '{index_name}' ({cloud}/{region})...")
    if args.recreate:
        log("Recreating index to clear old namespaces...")
    db = PineconeVectorDB(
        api_key=api_key,
        index_name=index_name,
        cloud=cloud,
        region=region,
        recreate=bool(args.recreate),
    )
    log(f"Index ready (single namespace 'questions', vector size {VECTOR_SIZE})")

    log("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    log("Reading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        log(f"Limited to first {len(df)} rows")
    else:
        log(f"Loaded {len(df)} rows")

    by_collection: dict[str, list[str]] = defaultdict(list)
    unique_docs: dict[str, dict] = {}
    skipped = 0

    log("Preparing documents...")
    for index, row in enumerate(df.to_dict(orient="records"), start=1):
        topic = raw(row.get("Question Topic"))
        subtopic = raw(row.get("Question Subtopic"))
        difficulty = raw(row.get("Question Difficulty"))
        content = raw(row.get("Question Content"))
        options = raw(row.get("Options Data"))
        page_content = build_page_content(topic, subtopic, difficulty, content, options)
        if not page_content.strip():
            skipped += 1
            continue

        question_id = raw(row.get("Question ID"))
        qid = normalize_question_id(question_id) or point_uuid(question_id)
        collection = collection_for_row(topic, options)
        unique_docs[qid] = {
            "id": point_uuid(question_id),
            "content": page_content,
            "metadata": {
                "question_id": question_id,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": difficulty,
                "unit_tag": raw(row.get("Unit Tag of Question")),
                "module_tag": raw(row.get("Module Tag of Question")),
                "course_tag": raw(row.get("Course Tag of Question")),
                "grit_tag": raw(row.get("Grit Tag of Question")),
                "extra_tags": raw(row.get("Extra Tags")),
            },
        }
        by_collection[collection].append(qid)
        if index % 10000 == 0:
            log(f"  prepared {index}/{len(df)}")

    log(f"Unique questions: {len(unique_docs)} | collections: {len(by_collection)} | skipped: {skipped}")

    qid_to_collection: dict[str, str] = {}
    for collection_name, qids in by_collection.items():
        for qid in qids:
            qid_to_collection[qid] = collection_name

    ordered_ids = list(unique_docs.keys())
    if args.resume and not args.recreate:
        log("Listing existing Pinecone IDs for resume...")
        existing_ids = list_existing_vector_ids(db)
        log(f"Already stored: {len(existing_ids)}")
        before = len(ordered_ids)
        ordered_ids = [qid for qid in ordered_ids if unique_docs[qid]["id"] not in existing_ids]
        log(f"Remaining to upload: {len(ordered_ids)} (skipped {before - len(ordered_ids)})")
        if not ordered_ids:
            log("Nothing left to upload. Done.")
            return

    uploaded = 0
    log("Embedding + uploading remaining questions to Pinecone cloud...")
    for start in range(0, len(ordered_ids), BATCH_SIZE):
        batch_ids = ordered_ids[start : start + BATCH_SIZE]
        texts = [unique_docs[qid]["content"] for qid in batch_ids]
        vectors = embeddings.embed_documents(texts)

        by_ns: dict[str, list[dict]] = defaultdict(list)
        for qid, vector in zip(batch_ids, vectors):
            record = {
                "id": unique_docs[qid]["id"],
                "values": vector,
                "page_content": unique_docs[qid]["content"],
                "metadata": unique_docs[qid]["metadata"],
            }
            by_ns[qid_to_collection[qid]].append(record)

        for collection_name, records in by_ns.items():
            db.upsert_records(collection_name, records)

        uploaded += len(batch_ids)
        if start == 0 or uploaded == len(ordered_ids) or (start // BATCH_SIZE) % 5 == 0:
            log(f"  uploaded {uploaded}/{len(ordered_ids)} remaining")

    counts = {name: len(qids) for name, qids in by_collection.items()}
    counts["all_questions"] = len(unique_docs)
    db.set_collection_counts(counts)

    stats = db.index.describe_index_stats()
    total = getattr(stats, "total_vector_count", None) or 0
    log("Done.")
    log(f"Pinecone total vectors now: {total}")
    log("Check Pinecone console index: topin-questions (namespace: questions)")
    log("Start app with: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc}")
        sys.exit(1)

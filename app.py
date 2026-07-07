import csv
import io
import json
import os
import re

import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

st.set_page_config(page_title="Topin Global Search", page_icon="🤖", layout="wide")
st.title("🤖 Topin Global Question Engine")
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
DEFAULT_RESULT_LIMIT = 15
MAX_RESULT_LIMIT = 500
ALL_FETCH_CAP = 2000
RESULTS_PER_PAGE = 20

SUBJECT_ALIASES = [
    ("html/css", "html_css"),
    ("html css", "html_css"),
    ("c++", "cpp"),
    ("c#", "csharp"),
    ("csharp", "csharp"),
    ("javascript", "js"),
    ("typescript", "js"),
    ("reactjs", "reactjs"),
    ("react.js", "reactjs"),
    ("nodejs", "nodejs"),
    ("node.js", "nodejs"),
    ("node js", "nodejs"),
    ("python", "python"),
    ("java", "java"),
    ("react", "react"),
    ("node", "nodejs"),
    ("php", "php"),
    ("sql", "sql"),
    ("cpp", "cpp"),
    ("html", "html_css"),
    ("css", "html_css"),
    ("pandas", "python"),
    ("numpy", "python"),
    ("matplotlib", "python"),
    ("selenium", "selenium"),
    ("testing", "testing"),
    ("git", "git"),
    ("web", "web"),
    ("ide", "ide"),
    ("dsa", "dsa"),
    ("c", "c"),
]

SUBJECT_OPTIONS = [
    ("python", "Python"),
    ("java", "Java"),
    ("js", "JavaScript"),
    ("cpp", "C++"),
    ("c", "C"),
    ("csharp", "C#"),
    ("react", "React"),
    ("reactjs", "ReactJS"),
    ("nodejs", "Node.js"),
    ("php", "PHP"),
    ("sql", "SQL"),
    ("git", "Git"),
    ("html_css", "HTML/CSS"),
    ("web", "Web"),
    ("dsa", "DSA"),
    ("testing", "Testing"),
    ("selenium", "Selenium"),
    ("ide", "IDE"),
]

QUESTION_TYPE_OPTIONS = [
    ("mixed", "Mixed (all types)"),
    ("coding", "Coding"),
    ("coding_analysis", "Coding Analysis"),
    ("mcq", "MCQ"),
]

COUNT_OPTIONS = [
    ("1", "1 question"),
    ("5", "5 questions"),
    ("10", "10 questions"),
    ("15", "15 questions"),
    ("20", "20 questions"),
    ("50", "50 questions"),
    ("all", "All questions"),
]

DIFFICULTY_OPTIONS = [
    ("any", "Any difficulty"),
    ("basic", "Basic / Easy"),
    ("medium", "Medium"),
    ("advanced", "Advanced / Hard"),
]

# Standalone library collections (e.g. topic_pandas) do not use _coding/_mcq suffixes.
STANDALONE_TOPIC_TYPES = {
    "topic_pandas": "coding",
    "topic_numpy": "coding",
    "topic_pandas_mcq": "mcq",
}

LIBRARY_TOPIC_COLLECTIONS = {
    "pandas": {
        "coding": ["topic_pandas"],
        "mcq": ["topic_pandas_mcq"],
        "coding_analysis": ["topic_pandas_mcq"],
    },
    "numpy": {
        "coding": ["topic_numpy"],
    },
}

SUBJECT_COLLECTION_PREFIXES = {
    "python": ["topic_python_"],
    "java": ["topic_java_"],
    "js": ["topic_js_"],
    "cpp": ["topic_cpp_"],
    "csharp": ["topic_csharp_"],
    "react": ["topic_react_", "topic_reactjs_"],
    "reactjs": ["topic_reactjs_", "topic_react_"],
    "nodejs": ["topic_nodejs_", "topic_node_"],
    "html_css": ["topic_html_css_"],
    "c": ["topic_c_"],
    "dsa": ["topic_dsa_"],
    "php": ["topic_php_"],
    "sql": ["topic_sql_"],
    "git": ["topic_git_"],
    "web": ["topic_web_"],
    "ide": ["topic_ide_"],
    "testing": ["topic_testing_"],
    "selenium": ["topic_java_selenium_"],
}

FIELD_PREFIXES = (
    ("Topic:", "topic"),
    ("Subtopic:", "subtopic"),
    ("Difficulty:", "difficulty"),
    ("Question Text:", "question_text"),
    ("Short Description:", "short_description"),
    ("Options:", "options"),
    ("Tags:", "tags"),
)

TAG_SOURCE_COLUMNS = (
    "Extra Tags",
    "Course Tag of Question",
    "Module Tag of Question",
    "Unit Tag of Question",
    "Grit Tag of Question",
)

# Additional candidate columns commonly used for pool/offline/status information
EXTRA_POTENTIAL_TAG_FIELDS = [
    "Question Offline Status",
    "Question Pool",
    "Pool",
    "Offline Status",
    "InOfflineExam",
    "Question In Offline Exam",
    "Question Offline",
    "Question Status",
    "Question Source",
    "Source",
    "Pool Tag",
    "Status",
]

DIFFICULTY_GROUPS = {
    "advanced": {"HARD", "DIFFICULT", "VERY HARD", "VERY_HARD"},
    "basic": {"EASY", "BASIC", "VERY EASY", "VERY_EASY"},
    "medium": {"MEDIUM", "MODERATE", "MIXED"},
}


def _raw_value(value: str) -> str:
    if not value or str(value).lower() in {"nan", "none"}:
        return ""
    return str(value).strip()


def normalize_question_id(qid: str) -> str:
    if not qid:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(qid).lower())


def parse_question_id_from_query(query: str) -> str | None:
    """Extract a 32-char question ID (with or without dashes) from natural-language queries."""
    dashed = re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        query,
        re.IGNORECASE,
    )
    if dashed:
        return normalize_question_id(dashed.group(0))

    labeled = re.search(
        r"\b(?:question\s+)?id[:\s#-]*([0-9a-f]{32})\b",
        query,
        re.IGNORECASE,
    )
    if labeled:
        return normalize_question_id(labeled.group(1))

    bare = re.search(r"(?<![0-9a-f])([0-9a-f]{32})(?![0-9a-f])", query, re.IGNORECASE)
    if bare:
        return normalize_question_id(bare.group(1))

    return None


def build_question_id_intent(question_id: str) -> dict:
    _, _, _, question_topics = load_question_tag_index()
    topic = question_topics.get(question_id)
    topics = [topic] if topic else []
    return {
        "question_id": question_id,
        "subject": None,
        "question_type": None,
        "subtopic": None,
        "subtopics": [],
        "subtopic_keywords": [],
        "topics": topics,
        "primary_topics": topics,
        "cross_topics": [],
        "subject_inferred": False,
        "tags": [],
        "difficulty": None,
        "mixed": False,
        "limit": 1,
        "fetch_all": False,
        "has_explicit_count": True,
        "has_explicit_difficulty": False,
    }


def fetch_question_hit_by_id(
    client,
    question_id: str,
    topics: list[str] | None = None,
) -> dict | None:
    """Fetch a single question payload from Qdrant by normalized question ID."""
    _, _, _, question_topics = load_question_tag_index()
    target = normalize_question_id(question_id)
    if not target:
        return None

    collections: list[str] = []
    if topics:
        collections = [f"{topic}_questions" for topic in topics if topic]
    elif target in question_topics:
        collections = [f"{question_topics[target]}_questions"]
    else:
        return None

    for collection_name in collections:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                content = payload.get("page_content", "")
                stored_id = normalize_question_id(metadata.get("question_id", ""))
                if not stored_id:
                    match = re.search(r"Question ID:\s*([^\n]+)", content)
                    stored_id = normalize_question_id(match.group(1)) if match else ""
                if stored_id == target:
                    return {
                        "score": 1.0,
                        "content": content,
                        "collection": collection_name,
                        "metadata": metadata,
                    }

            if offset is None:
                break

    return None


def normalize_tag_key(tag: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(tag).upper())

data_link=st.secrets.get("data_link")
@st.cache_data
def load_question_tag_index():
    # Read CSV header first to detect which tag-like columns actually exist
    header_df = pd.read_csv(data_link, nrows=0)
    available_cols = set(header_df.columns.tolist())

    # Build effective usecols: always include Question ID and Question Topic
    usecols = ["Question ID", "Question Topic"]
    for col in list(TAG_SOURCE_COLUMNS) + EXTRA_POTENTIAL_TAG_FIELDS:
        if col in available_cols and col not in usecols:
            usecols.append(col)

    df = pd.read_csv(
        data_link,
        usecols=usecols,
        low_memory=False,
    )
    tag_index: dict[str, set[str]] = {}
    tag_display: dict[str, dict[str, str]] = {}
    tag_to_questions: dict[str, set[str]] = {}
    question_topics: dict[str, str] = {}

    for _, row in df.iterrows():
        qid = normalize_question_id(str(row["Question ID"]))
        topic = _raw_value(row.get("Question Topic", ""))
        if topic:
            question_topics[qid] = topic

        tokens: set[str] = set()
        for col in TAG_SOURCE_COLUMNS:
            raw = _raw_value(row.get(col, ""))
            if not raw:
                continue
            tokens.add(raw.upper().replace(" ", "_"))
            for token in raw.split(","):
                cleaned = token.strip().upper().replace(" ", "_")
                if cleaned and cleaned != "NAN":
                    tokens.add(cleaned)

        for col in EXTRA_POTENTIAL_TAG_FIELDS:
            raw = _raw_value(row.get(col, ""))
            if not raw:
                continue
            tokens.add(raw.upper().replace(" ", "_"))
            for token in raw.split(","):
                cleaned = token.strip().upper().replace(" ", "_")
                if cleaned and cleaned != "NAN":
                    tokens.add(cleaned)

        tag_index[qid] = tokens
        for token in tokens:
            tag_to_questions.setdefault(token, set()).add(qid)

        tag_display[qid] = {
            "extra_tags": _raw_value(row.get("Extra Tags", "")),
            "course_tag": _raw_value(row.get("Course Tag of Question", "")),
            "module_tag": _raw_value(row.get("Module Tag of Question", "")),
            "unit_tag": _raw_value(row.get("Unit Tag of Question", "")),
            "grit_tag": _raw_value(row.get("Grit Tag of Question", "")),
            "all_tags": ", ".join(sorted(tokens)),
        }

    return tag_index, tag_display, tag_to_questions, question_topics


def canonicalize_tags(required_tags: list[str], tag_to_questions: dict[str, set[str]]) -> list[str]:
    canonical: list[str] = []
    for tag in required_tags:
        upper = str(tag).upper().replace(" ", "_").strip()
        if upper in tag_to_questions:
            canonical.append(upper)
            continue
        norm = normalize_tag_key(upper)
        matched = next(
            (stored for stored in tag_to_questions if normalize_tag_key(stored) == norm),
            None,
        )
        if matched:
            canonical.append(matched)
            continue
        fuzzy = fuzzy_match_tag_to_catalog(tag, tag_to_questions)
        canonical.append(fuzzy or upper)
    return list(dict.fromkeys(canonical))


def fuzzy_match_tag_to_catalog(tag: str, tag_to_questions: dict[str, set[str]]) -> str | None:
    """Map natural tag text (with or without underscores) to a stored CSV tag."""
    if not tag:
        return None

    upper = str(tag).upper().replace(" ", "_").strip()
    if upper in tag_to_questions:
        return upper

    norm = normalize_tag_key(upper)
    for stored in tag_to_questions:
        if normalize_tag_key(stored) == norm:
            return stored

    user_tokens = [token for token in _normalize_tag_text(tag).split("_") if token]
    if not user_tokens:
        return None

    candidates: list[tuple[str, int]] = []
    for stored in tag_to_questions:
        stored_tokens = [token for token in _normalize_tag_text(stored).split("_") if token]
        if not stored_tokens:
            continue
        if user_tokens == stored_tokens:
            return stored
        if all(token in stored_tokens for token in user_tokens):
            candidates.append((stored, len(stored_tokens)))

    if not candidates:
        return None

    best_len = max(length for _, length in candidates)
    best = [stored for stored, length in candidates if length == best_len]
    return best[0] if len(best) == 1 else None


def resolve_tag_search(required_tags: list[str]) -> tuple[set[str], list[str], list[str]]:
    """Map requested tags to question IDs and optional topic collections (hints only)."""
    tag_index, _, tag_to_questions, question_topics = load_question_tag_index()
    canonical_tags = canonicalize_tags(required_tags, tag_to_questions)
    matched_ids: set[str] = set()
    for tag in canonical_tags:
        matched_ids |= tag_to_questions.get(tag, set())

    collections = sorted(
        {
            f"{question_topics[qid]}_questions"
            for qid in matched_ids
            if qid in question_topics
        }
    )
    return matched_ids, collections, canonical_tags


def is_tag_primary_intent(intent: dict) -> bool:
    """Tag-only search: return every question with the tag, regardless of Question Topic."""
    return bool(intent.get("tag_primary") or (intent.get("tags") and not intent.get("topics")))


def intent_without_topic_scope(intent: dict) -> dict:
    """Intent copy that keeps type/subject filters but drops topic narrowing."""
    scoped = dict(intent)
    for key in ("topics", "primary_topics", "subtopic", "subtopics", "subtopic_keywords"):
        scoped.pop(key, None)
    return scoped


@st.cache_data
def load_csv_questions_by_id() -> dict[str, pd.Series]:
    df = pd.read_csv(data_link, low_memory=False)
    by_id: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        qid = normalize_question_id(str(row.get("Question ID", "")))
        if qid:
            by_id[qid] = row
    return by_id


def build_hit_from_csv_row(row: pd.Series) -> dict:
    topic = _raw_value(row.get("Question Topic", ""))
    subtopic = _raw_value(row.get("Question Subtopic", ""))
    difficulty = _raw_value(row.get("Question Difficulty", ""))
    question_id = _raw_value(row.get("Question ID", ""))
    options = _raw_value(row.get("Options Data", ""))
    has_options = bool(options and options.lower() not in {"nan", "none", "[]", "{}"})
    if topic:
        collection = f"{topic}_questions"
    elif has_options:
        collection = "unassigned_mcq_questions"
    else:
        collection = "unassigned_coding_questions"

    lines: list[str] = []
    if topic:
        lines.append(f"Topic: {topic}")
    if subtopic:
        lines.append(f"Subtopic: {subtopic}")
    if difficulty:
        lines.append(f"Difficulty: {difficulty}")
    content = _raw_value(row.get("Question Content", ""))
    if content:
        lines.append(f"Question Text:\n{content}")
    if options:
        lines.append(f"Options: {options}")

    metadata = {
        "question_id": question_id,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "unit_tag": _raw_value(row.get("Unit Tag of Question", "")),
        "module_tag": _raw_value(row.get("Module Tag of Question", "")),
        "course_tag": _raw_value(row.get("Course Tag of Question", "")),
        "grit_tag": _raw_value(row.get("Grit Tag of Question", "")),
        "extra_tags": _raw_value(row.get("Extra Tags", "")),
    }
    return {
        "score": 1.0,
        "content": "\n".join(lines),
        "collection": collection,
        "metadata": metadata,
    }


def build_csv_fallback_hits(matched_ids: set[str], found_ids: set[str]) -> list[dict]:
    missing = matched_ids - found_ids
    if not missing:
        return []
    by_id = load_csv_questions_by_id()
    return [build_hit_from_csv_row(by_id[qid]) for qid in sorted(missing) if qid in by_id]


TAG_SCROLL_BATCH = 256


def _extract_qid_from_point(point) -> str:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return _extract_qid_from_point_payload(metadata, payload.get("page_content", ""))


def _extract_qid_from_point_payload(metadata: dict, content: str) -> str:
    qid = normalize_question_id(metadata.get("question_id", ""))
    if not qid:
        match = re.search(r"Question ID:\s*([^\n]+)", content)
        qid = normalize_question_id(match.group(1)) if match else ""
    return qid


def _fetch_ids_from_collection(client, collection_name: str, needed_ids: set[str]) -> list[dict]:
    if not needed_ids:
        return []

    remaining = set(needed_ids)
    hits: list[dict] = []
    offset = None

    while remaining:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=TAG_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            qid = _extract_qid_from_point(point)
            if qid not in remaining:
                continue
            remaining.discard(qid)
            payload = point.payload or {}
            hits.append(
                {
                    "score": 1.0,
                    "content": payload.get("page_content", ""),
                    "collection": collection_name,
                    "metadata": payload.get("metadata", {}) or {},
                }
            )

        if offset is None:
            break

    return hits


def _fetch_tag_primary_hits(client, matched_ids: set[str]) -> list[dict]:
    """Fast tag lookup: only scan collections known from CSV, CSV-only for questions without topic."""
    if not matched_ids:
        return []

    _, _, _, question_topics = load_question_tag_index()
    qdrant_ids = {qid for qid in matched_ids if qid in question_topics}
    csv_only_ids = matched_ids - qdrant_ids

    collection_targets: dict[str, set[str]] = {}
    for qid in qdrant_ids:
        collection_targets.setdefault(f"{question_topics[qid]}_questions", set()).add(qid)

    hits: list[dict] = []
    if collection_targets:
        max_workers = min(8, len(collection_targets))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_fetch_ids_from_collection, client, collection_name, needed_ids)
                for collection_name, needed_ids in collection_targets.items()
            ]
            for future in as_completed(futures):
                hits.extend(future.result())

    if csv_only_ids:
        hits.extend(build_csv_fallback_hits(csv_only_ids, set()))

    found_ids = {
        qid
        for hit in hits
        for qid in [_extract_qid_from_point_payload(hit.get("metadata") or {}, hit.get("content", ""))]
        if qid
    }
    missing_qdrant = qdrant_ids - found_ids
    if missing_qdrant:
        hits.extend(build_csv_fallback_hits(missing_qdrant, set()))

    return hits


@st.cache_data(ttl=900)
def fetch_cached_tag_primary_hits(tag_signature: str, matched_ids_tuple: tuple[str, ...]) -> list[dict]:
    client = load_qdrant_client()
    return _fetch_tag_primary_hits(client, set(matched_ids_tuple))


SUBTOPIC_QUERY_STOPWORDS = {
    "give",
    "show",
    "get",
    "list",
    "fetch",
    "display",
    "find",
    "want",
    "need",
    "question",
    "questions",
    "mcq",
    "mcqs",
    "problem",
    "problems",
    "coding",
    "analysis",
    "mixed",
    "topic",
    "easy",
    "basic",
    "medium",
    "hard",
    "advanced",
    "all",
    "every",
    "the",
    "and",
    "for",
    "from",
    "with",
    "about",
    "only",
    "just",
    "more",
    "content",
    "existing",
    "overall",
    "commands",
    "command",
    "material",
    "coverage",
    "available",
    "everything",
}

EXPLORE_FILLER_KEYWORDS = SUBTOPIC_QUERY_STOPWORDS


def _subtopic_keyword_variants(suffix: str) -> set[str]:
    variants = {suffix}
    for part in re.split(r"[_\s&/]+", suffix):
        cleaned = part.strip().lower()
        if len(cleaned) >= 3 and cleaned not in SUBTOPIC_QUERY_STOPWORDS:
            variants.add(cleaned)
    return variants


@st.cache_data
def load_subtopic_index():
    df = pd.read_csv(
        data_link,
        usecols=["Question Topic", "Question Subtopic"],
        low_memory=False,
    )
    keyword_map: dict[str, list[dict]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}

    for _, row in df.iterrows():
        topic = _raw_value(row.get("Question Topic", ""))
        subtopic = _raw_value(row.get("Question Subtopic", "")).upper()
        if not topic or not subtopic:
            continue

        entry = {"topic": topic, "subtopic": subtopic}
        suffix = subtopic.replace("SUB_TOPIC_", "").lower()
        for keyword in _subtopic_keyword_variants(suffix):
            if keyword in SUBTOPIC_QUERY_STOPWORDS:
                continue
            bucket = seen.setdefault(keyword, set())
            key = (topic, subtopic)
            if key in bucket:
                continue
            bucket.add(key)
            keyword_map.setdefault(keyword, []).append(entry)

    return keyword_map


@st.cache_data
def load_topic_index():
    """Map query keywords to question topics/collections from CSV topic names (e.g. terraform -> topic_terraform_mcq)."""
    df = pd.read_csv(
        data_link,
        usecols=["Question Topic"],
        low_memory=False,
    )
    keyword_map: dict[str, set[str]] = {}

    for raw_topic in df["Question Topic"].dropna().unique():
        topic = _raw_value(raw_topic)
        if not topic:
            continue

        body = topic[6:] if topic.startswith("topic_") else topic
        body_lower = body.lower()
        keyword_map.setdefault(body_lower, set()).add(topic)

        for segment in body_lower.split("_"):
            if len(segment) >= 3 and segment not in SUBTOPIC_QUERY_STOPWORDS:
                keyword_map.setdefault(segment, set()).add(topic)

        for suffix in ("_coding_analysis", "_coding", "_mcq"):
            if body_lower.endswith(suffix):
                stem = body_lower[: -len(suffix)]
                if len(stem) >= 3:
                    keyword_map.setdefault(stem, set()).add(topic)
                break

    return {keyword: sorted(topics) for keyword, topics in keyword_map.items()}


def extract_matched_keywords_from_query(query: str, keyword_map: dict) -> list[str]:
    """Match catalog keywords from spaced or underscore-separated queries (e.g. topic_stl_mcqs)."""
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    matched: list[str] = []

    for token in re.split(r"[_\s]+", normalized):
        if len(token) >= 3 and token not in SUBTOPIC_QUERY_STOPWORDS:
            if token in keyword_map and token not in matched:
                matched.append(token)

    for keyword in sorted(keyword_map.keys(), key=len, reverse=True):
        if len(keyword) < 3 or keyword in SUBTOPIC_QUERY_STOPWORDS:
            continue
        pattern = rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])"
        if re.search(pattern, normalized) and keyword not in matched:
            matched.append(keyword)

    return matched


def extract_topic_keywords_from_query(query: str, keyword_map: dict[str, list[str]]) -> list[str]:
    return extract_matched_keywords_from_query(query, keyword_map)


def resolve_topics_from_query(query: str, question_type: str | None = None) -> dict:
    keyword_map = load_topic_index()
    keywords = extract_topic_keywords_from_query(query, keyword_map)
    if not keywords:
        return {}

    topics: set[str] = set()
    for keyword in keywords:
        topics.update(keyword_map.get(keyword, []))

    if question_type:
        topics = {topic for topic in topics if topic_matches_question_type(topic, question_type)}

    if not topics:
        return {"topic_keywords": keywords, "topics": []}

    return {"topic_keywords": keywords, "topics": sorted(topics)}


@st.cache_data
def load_topic_catalog() -> list[str]:
    df = pd.read_csv(data_link, usecols=["Question Topic"], low_memory=False)
    return sorted({_raw_value(topic) for topic in df["Question Topic"].dropna().unique() if _raw_value(topic)})


def load_topic_catalog_set() -> set[str]:
    return set(load_topic_catalog())


TOPIC_SHORTHAND_PATTERN = re.compile(
    r"(?<![a-z0-9_])(topic_[a-z0-9]+(?:_[a-z0-9]+)*?)_(mcqs?|coding_analysis|coding)(?![a-z0-9_])",
    re.IGNORECASE,
)


def resolve_topic_shorthand(query: str) -> dict:
    """Parse collection-style names such as topic_stl_mcqs or topic_terraform_mcq."""
    match = TOPIC_SHORTHAND_PATTERN.search(query.lower())
    if not match:
        return {}

    topic_part = match.group(1).lower()
    type_suffix = match.group(2).lower()
    question_type = "mcq" if type_suffix.startswith("mcq") else type_suffix

    catalog = load_topic_catalog_set()
    if topic_part in catalog:
        return {
            "topics": [topic_part],
            "primary_topics": [topic_part],
            "question_type": question_type,
        }

    stem = topic_part[6:] if topic_part.startswith("topic_") else topic_part
    keywords = [
        part
        for part in stem.split("_")
        if len(part) >= 3 and part not in SUBTOPIC_QUERY_STOPWORDS
    ]
    if not keywords and len(stem) >= 3:
        keywords = [stem]

    return {
        "topic_keywords": keywords,
        "subtopic_keywords": keywords,
        "question_type": question_type,
    }


def infer_question_type_from_topics(topics: list[str]) -> str | None:
    types: set[str] = set()
    for topic in topics:
        coll_type = collection_question_type(topic)
        if coll_type:
            types.add(coll_type)
    if len(types) == 1:
        return next(iter(types))
    return None


def infer_question_type_from_query(query: str) -> str | None:
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    if re.search(r"\bmixed\b", normalized):
        return None
    if re.search(r"coding\s*analysis|code\s*analysis|coding_analysis", normalized):
        return "coding_analysis"
    if re.search(r"\b(mcq|mcqs|multiple\s*choice)\b", normalized):
        return "mcq"
    if re.search(r"\b(coding|code\s*question|write\s+(a\s+)?(program|code|function))\b", normalized):
        return "coding"
    return None


def is_comprehensive_content_query(query: str) -> bool:
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    patterns = [
        r"\b(overall|existing|entire|complete|full|all)\b.*\b(content|questions?|material|mcqs?)\b",
        r"\b(existing|overall)\s+content\b",
        r"\b(give|show|list|get|fetch)\b.*\b(overall|existing)\b.*\bcontent\b",
        r"\b(everything|all)\s+(on|about|related\s+to)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def clean_subtopic_keywords(keywords: list[str], subject: str | None, query: str) -> list[str]:
    cleaned = [keyword for keyword in keywords if keyword not in EXPLORE_FILLER_KEYWORDS]
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    if subject and re.search(rf"\b{re.escape(subject)}\s+commands?\b", normalized):
        cleaned = [keyword for keyword in cleaned if keyword not in {"commands", "command"}]
    if subject:
        cleaned = [subject] + [keyword for keyword in cleaned if keyword != subject]
    return list(dict.fromkeys(cleaned))


def refresh_keyword_scope(intent: dict, query: str) -> dict:
    """Re-resolve topics/subtopics from core keywords only (drops filler like 'content', 'commands')."""
    subject = intent.get("subject")
    keywords = clean_subtopic_keywords(intent.get("subtopic_keywords") or [], subject, query)
    if not keywords and subject:
        keywords = [subject]
    if not keywords:
        return intent

    intent = dict(intent)
    intent["subtopic_keywords"] = keywords
    synthetic = " ".join(keywords)
    subtopic_info = resolve_subtopics_from_query(synthetic, intent.get("question_type"))
    topic_info = resolve_topics_from_query(synthetic, intent.get("question_type"))
    topics_set = set(subtopic_info.get("topics") or [])
    topics_set.update(topic_info.get("topics") or [])

    primary = sorted(topic for topic in topics_set if subject and _subject_matches_collection(topic, subject))
    if not primary:
        primary = sorted(
            topic for topic in topics_set
            if any(_topic_name_matches_keyword(topic, keyword) for keyword in keywords)
        )

    if topics_set:
        intent["topics"] = sorted(topics_set)
        intent["primary_topics"] = primary
        intent["cross_topics"] = sorted(topic for topic in topics_set if topic not in set(primary))
        intent["subtopics"] = sorted(
            subtopic
            for subtopic in (subtopic_info.get("subtopics") or [])
            if any(_subtopic_matches_keywords(subtopic, keywords) for keyword in keywords)
        )
    return intent


def finalize_query_intent(intent: dict, query: str) -> dict:
    """Normalize shorthand queries, infer missing filters, and enrich topic resolution."""
    intent = dict(intent)
    shorthand = resolve_topic_shorthand(query)

    if shorthand.get("question_type") and not intent.get("question_type") and not intent.get("mixed"):
        intent["question_type"] = shorthand["question_type"]
    if shorthand.get("topics") and not intent.get("tag_primary"):
        intent["topics"] = shorthand["topics"]
        intent["primary_topics"] = shorthand.get("primary_topics", shorthand["topics"])
    elif shorthand.get("subtopic_keywords") and not intent.get("tag_primary"):
        merged_keywords = list(intent.get("subtopic_keywords") or [])
        for keyword in shorthand["subtopic_keywords"]:
            if keyword not in merged_keywords:
                merged_keywords.append(keyword)
        intent["subtopic_keywords"] = merged_keywords

    if not intent.get("question_type") and not intent.get("mixed"):
        intent["question_type"] = infer_question_type_from_query(query)

    if not intent.get("tag_primary") and not intent.get("topics"):
        subtopic_info = resolve_subtopics_from_query(query, intent.get("question_type"))
        topic_info = resolve_topics_from_query(query, intent.get("question_type"))
        topics_set = set(subtopic_info.get("topics") or [])
        topics_set.update(topic_info.get("topics") or [])
        if topics_set:
            keywords = list(intent.get("subtopic_keywords") or [])
            for keyword in subtopic_info.get("subtopic_keywords") or []:
                if keyword not in keywords:
                    keywords.append(keyword)
            for keyword in topic_info.get("topic_keywords") or []:
                if keyword not in keywords:
                    keywords.append(keyword)
            intent["subtopic_keywords"] = keywords
            intent["topics"] = sorted(topics_set)
            intent["primary_topics"] = list(intent["topics"])
            if subtopic_info.get("subtopics"):
                intent["subtopics"] = subtopic_info["subtopics"]

    if (
        not intent.get("tag_primary")
        and not intent.get("question_type")
        and not intent.get("mixed")
        and intent.get("topics")
    ):
        inferred = infer_question_type_from_topics(intent["topics"])
        if inferred:
            intent["question_type"] = inferred

    if intent.get("tag_primary") or has_curriculum_tags(intent.get("tags") or []):
        intent["subtopic_keywords"] = []
        intent["topics"] = []
        intent["primary_topics"] = []
        intent["subtopics"] = []
        intent["subtopic"] = None
    else:
        intent = apply_tag_hints_to_intent(intent)

    intent = refresh_keyword_scope(intent, query)

    if is_comprehensive_content_query(query):
        intent["fetch_all"] = True
        intent["limit"] = None
        intent["has_explicit_count"] = True

    return sanitize_intent_topics(intent)


def resolve_curriculum_tags_from_keywords(keywords: list[str]) -> list[str]:
    """Map shorthand keywords (e.g. stl) to MODULE_/UNIT_/COURSE_ tags in the CSV index."""
    _, _, tag_to_questions, _ = load_question_tag_index()
    found: list[str] = []
    for keyword in keywords:
        kw_key = normalize_tag_key(keyword)
        if len(kw_key) < 3:
            continue
        for tag in tag_to_questions:
            if not tag.startswith(("MODULE_", "UNIT_", "COURSE_")):
                continue
            tag_key = normalize_tag_key(tag)
            tag_tokens = re.sub(r"[^A-Z0-9]+", "_", tag.upper()).strip("_").split("_")
            if kw_key in {normalize_tag_key(token) for token in tag_tokens if token}:
                found.append(tag)
            elif tag_key.endswith(f"_{kw_key}") or tag_key.startswith(f"{kw_key}_"):
                found.append(tag)

    found.sort(key=lambda tag: (0 if tag.startswith("MODULE_") else 1, len(tag)))
    return list(dict.fromkeys(found))


def apply_tag_hints_to_intent(intent: dict) -> dict:
    if intent.get("tags") or has_curriculum_tags(intent.get("tags") or []):
        return intent
    if intent.get("subject"):
        keywords = intent.get("subtopic_keywords") or []
        if not keywords or intent["subject"] in keywords:
            return intent

    tag_candidates = resolve_curriculum_tags_from_keywords(intent.get("subtopic_keywords") or [])
    if not tag_candidates:
        return intent

    intent = dict(intent)
    intent["tags"] = [tag_candidates[0]]
    _matched_ids, tag_collections, canonical_tags = resolve_tag_search(intent["tags"])
    intent["tags"] = canonical_tags
    if tag_collections:
        derived_topics = sorted(name.removesuffix("_questions") for name in tag_collections)
        if intent.get("topics"):
            intent["topics"] = sorted(set(intent["topics"]) | set(derived_topics))
        else:
            intent["topics"] = derived_topics
            intent["primary_topics"] = list(derived_topics)
    return intent


def filter_topics_for_intent(topics: list[str], intent: dict) -> list[str]:
    return [topic for topic in topics if topic_matches_intent_scope(topic, intent)]


def topic_matches_intent_scope(topic: str, intent: dict) -> bool:
    if topic in (intent.get("cross_topics") or []):
        return True

    if intent.get("tags"):
        return True

    subject = intent.get("subject")
    keywords = intent.get("subtopic_keywords") or []

    if subject and _subject_matches_collection(topic, subject):
        return True
    if keywords:
        return any(_topic_name_matches_keyword(topic, keyword) for keyword in keywords)
    if subject:
        return _subject_matches_collection(topic, subject)
    return True


def get_trusted_collections(intent: dict) -> set[str]:
    trusted: set[str] = set()
    for topic in intent.get("primary_topics") or []:
        if topic_matches_intent_scope(topic, intent):
            trusted.add(topic.removesuffix("_questions"))

    subject = intent.get("subject")
    keywords = intent.get("subtopic_keywords") or []
    for topic in intent.get("topics") or []:
        base = topic.removesuffix("_questions")
        if subject and _subject_matches_collection(base, subject):
            trusted.add(base)
            continue
        if keywords and any(_topic_name_matches_keyword(base, keyword) for keyword in keywords):
            trusted.add(base)
    return trusted


def sanitize_intent_topics(intent: dict) -> dict:
    intent = dict(intent)
    if intent.get("topics"):
        intent["topics"] = filter_topics_for_intent(intent["topics"], intent)
    if intent.get("primary_topics"):
        intent["primary_topics"] = filter_topics_for_intent(intent["primary_topics"], intent)
    elif intent.get("topics"):
        intent["primary_topics"] = [
            topic for topic in intent["topics"] if topic in get_trusted_collections(intent)
        ]
    return intent


@st.cache_data
def load_sample_tags(limit: int = 80) -> list[str]:
    _, _, tag_to_questions, _ = load_question_tag_index()
    grit = sorted(tag for tag in tag_to_questions if tag.startswith("GRIT_"))
    curriculum = sorted(
        tag for tag in tag_to_questions if tag.startswith(("UNIT_", "MODULE_", "COURSE_"))
    )
    other = sorted(
        tag for tag in tag_to_questions if not tag.startswith(("GRIT_", "UNIT_", "MODULE_", "COURSE_", "SET_", "WEEK_"))
    )
    combined = list(dict.fromkeys(grit + curriculum + other))
    return combined[:limit]


def build_llm_intent_catalog() -> str:
    return json.dumps(
        {
            "topics": load_topic_catalog(),
            "subjects": [value for value, _label in SUBJECT_OPTIONS],
            "question_types": ["mcq", "coding", "coding_analysis", "mixed"],
            "difficulties": ["basic", "medium", "advanced"],
            "tag_prefixes": ["UNIT_", "MODULE_", "COURSE_", "GRIT_", "SET_", "WEEK_"],
            "sample_tags": load_sample_tags(),
            "examples": [
                "topic_stl_mcqs -> subtopic_keywords ['stl'], question_type 'mcq'",
                "give 10 easy terraform mcqs -> topics ['topic_terraform_mcq'], difficulty 'basic'",
                "UNIT_INTRODUCTION_TO_STACKS MCQs -> tags ['UNIT_INTRODUCTION_TO_STACKS'], question_type 'mcq'",
                "GRIT_CS_FUNDAMENTALS_MAIN_L2 -> tags ['GRIT_CS_FUNDAMENTALS_MAIN_L2']",
            ],
        },
        indent=2,
    )


def topic_matches_question_type(topic: str, question_type: str | None) -> bool:
    if not question_type:
        return True
    standalone_type = STANDALONE_TOPIC_TYPES.get(topic)
    if standalone_type:
        return question_type == standalone_type
    if question_type == "coding_analysis":
        return "_coding_analysis" in topic
    if question_type == "coding":
        return "_coding" in topic and "_coding_analysis" not in topic
    if question_type == "mcq":
        return "_mcq" in topic
    return True


def resolve_library_topics_from_query(query: str, question_type: str | None) -> dict:
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    for library, type_map in LIBRARY_TOPIC_COLLECTIONS.items():
        if not re.search(rf"\b{re.escape(library)}\b", normalized):
            continue
        if question_type and question_type in type_map:
            topics = list(type_map[question_type])
        elif question_type == "mixed":
            topics = sorted({topic for topics in type_map.values() for topic in topics})
        else:
            topics = sorted({topic for topics in type_map.values() for topic in topics})
        return {"library": library, "topics": topics}
    return {"library": None, "topics": []}


def collection_question_type(base: str) -> str | None:
    if base in STANDALONE_TOPIC_TYPES:
        return STANDALONE_TOPIC_TYPES[base]
    if "_coding_analysis" in base:
        return "coding_analysis"
    if "_coding" in base:
        return "coding"
    if "_mcq" in base:
        return "mcq"
    return None


def topic_to_subject(topic: str) -> str | None:
    if not topic.startswith("topic_"):
        return None

    known_subjects = {subject for subject, _label in SUBJECT_OPTIONS}
    body = topic[6:]
    first = body.split("_")[0]
    if first in known_subjects:
        return first

    for subject in sorted(known_subjects, key=len, reverse=True):
        if body.startswith(f"{subject}_"):
            return subject

    if body.startswith("pandas") or body.startswith("numpy"):
        return "python"
    return None


def extract_subtopic_keywords_from_query(query: str, keyword_map: dict[str, list[dict]]) -> list[str]:
    return extract_matched_keywords_from_query(query, keyword_map)


def resolve_subtopics_from_query(query: str, question_type: str | None = None) -> dict:
    keyword_map = load_subtopic_index()
    keywords = extract_subtopic_keywords_from_query(query, keyword_map)
    if not keywords:
        return {}

    topics: set[str] = set()
    subtopics: set[str] = set()
    for keyword in keywords:
        for entry in keyword_map.get(keyword, []):
            topics.add(entry["topic"])
            subtopics.add(entry["subtopic"])

    if question_type:
        topics = {topic for topic in topics if topic_matches_question_type(topic, question_type)}

    if not topics:
        return {
            "subtopic_keywords": keywords,
            "subtopics": sorted(subtopics),
            "topics": [],
            "subject": None,
        }

    subjects = {topic_to_subject(topic) for topic in topics}
    subjects.discard(None)
    inferred_subject = next(iter(subjects)) if len(subjects) == 1 else None

    return {
        "subtopic_keywords": keywords,
        "subtopics": sorted(subtopics),
        "topics": sorted(topics),
        "subject": inferred_subject,
    }


def _keyword_matches_token(text: str, keyword: str) -> bool:
    """Match whole underscore tokens so 'git' does not match 'github' or unrelated labels."""
    if not text or not keyword:
        return False
    text_tokens = {
        token
        for token in re.sub(r"[^a-z0-9]+", "_", text.lower()).split("_")
        if token and token not in SUBTOPIC_QUERY_STOPWORDS
    }
    keyword_token = re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
    if not keyword_token:
        return False
    return keyword_token in text_tokens


def _subtopic_matches_keywords(subtopic: str, keywords: list[str]) -> bool:
    subtopic_upper = subtopic.upper()
    subtopic_body = subtopic_upper.replace("SUB_TOPIC_", "")
    for keyword in keywords:
        token = keyword.upper()
        if _keyword_matches_token(subtopic_body, keyword):
            return True
        if _keyword_matches_token(subtopic_upper, keyword):
            return True
        if token == subtopic_body or token == subtopic_upper:
            return True
    return False


def _topic_name_matches_keyword(topic: str, keyword: str) -> bool:
    return _keyword_matches_token(topic, keyword)


def enrich_cross_collection_intent(
    keywords: list[str],
    topics: list[str],
    all_subtopics: list[str],
    subject: str | None,
    question_type: str | None,
    library_info: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    For any matched keyword, search every CSV-indexed collection for that keyword.
    Primary collections (name contains keyword) are trusted as-is; other collections
    require the question subtopic field to match the keyword.
    """
    if not keywords or not topics:
        return topics, [], all_subtopics

    topic_set = set(topics)
    primary: set[str] = set()

    if library_info and library_info.get("topics"):
        primary.update(library_info["topics"])

    for topic in topic_set:
        for keyword in keywords:
            if _topic_name_matches_keyword(topic, keyword):
                primary.add(topic)
                break

    if subject:
        subject_prefix = f"topic_{subject}_"
        for topic in topic_set:
            if not topic.startswith(subject_prefix):
                continue
            for keyword in keywords:
                if _topic_name_matches_keyword(topic, keyword):
                    primary.add(topic)
                    break

    cross_topics = topic_set - primary
    if not cross_topics:
        return sorted(topic_set), sorted(primary), []

    filter_subtopics = sorted(
        subtopic for subtopic in all_subtopics if _subtopic_matches_keywords(subtopic, keywords)
    )
    if not filter_subtopics:
        filter_subtopics = sorted(all_subtopics)

    return sorted(topic_set), sorted(primary), filter_subtopics


STRUCTURED_TAG_PATTERN = re.compile(
    r"\b((?:UNIT|MODULE|COURSE|GRIT|SET|WEEK)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)\b",
    re.IGNORECASE,
)


def _normalize_tag_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def find_matching_tags_in_query(query: str, tag_to_questions: dict[str, set[str]]) -> list[str]:
    """Return the best-matching catalog tag(s) mentioned in the query."""
    normalized_query = _normalize_tag_text(query)
    if not normalized_query:
        return []

    substring_hits: list[tuple[str, int]] = []
    for stored in tag_to_questions:
        norm_tag = _normalize_tag_text(stored)
        if len(norm_tag) < 8:
            continue
        if re.search(rf"(?:^|_){re.escape(norm_tag)}(?:_|$)", normalized_query):
            substring_hits.append((stored, len(norm_tag)))

    if substring_hits:
        max_len = max(length for _, length in substring_hits)
        return list(dict.fromkeys(tag for tag, length in substring_hits if length == max_len))

    query_tokens = [token for token in normalized_query.split("_") if token]
    if len(query_tokens) < 2:
        return []

    token_hits: list[tuple[str, int]] = []
    for stored in tag_to_questions:
        tag_tokens = [token for token in _normalize_tag_text(stored).split("_") if token]
        if len(tag_tokens) < 2:
            continue
        if all(token in query_tokens for token in tag_tokens):
            token_hits.append((stored, len(tag_tokens)))

    if not token_hits:
        return []

    max_len = max(length for _, length in token_hits)
    return list(dict.fromkeys(tag for tag, length in token_hits if length == max_len))


def _query_matches_existing_tag(query: str, stored_tag: str) -> bool:
    _, _, tag_to_questions, _ = load_question_tag_index()
    return stored_tag in find_matching_tags_in_query(query, tag_to_questions)


def strip_structured_tags(query: str) -> str:
    return re.sub(r"\s+", " ", STRUCTURED_TAG_PATTERN.sub(" ", query)).strip()


def has_curriculum_tags(tags: list[str]) -> bool:
    return bool([tag for tag in tags if isinstance(tag, str) and tag.strip()])


def parse_query_tags(query: str) -> list[str]:
    tags: list[str] = []
    for match in STRUCTURED_TAG_PATTERN.finditer(query):
        tags.append(match.group(1).upper().replace(" ", "_"))
    for match in re.finditer(r"\bSET[_\s-]?(\d+)\b", query, re.IGNORECASE):
        tags.append(f"SET_{match.group(1)}")
    for match in re.finditer(r"\bWEEK[_\s-]?(\d+)\b", query, re.IGNORECASE):
        tags.append(f"WEEK_{match.group(1)}")
    for match in re.finditer(r"\btag[s]?\s+([A-Z0-9_&]+)", query, re.IGNORECASE):
        tags.append(match.group(1).upper().replace(" ", "_"))

    _, _, tag_to_questions, _ = load_question_tag_index()
    tags.extend(find_matching_tags_in_query(query, tag_to_questions))
    return list(dict.fromkeys(canonicalize_tags(tags, tag_to_questions)))


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def get_openrouter_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return key or None


def _normalize_llm_intent(raw: dict) -> dict:
    subjects = {value for value, _label in SUBJECT_OPTIONS}
    normalized = {
        "subject": raw.get("subject"),
        "question_type": raw.get("question_type"),
        "difficulty": raw.get("difficulty"),
        "count": raw.get("count"),
        "fetch_all": bool(raw.get("fetch_all")),
        "tags": [
            str(t).replace(" ", "_").strip()
            for t in (raw.get("tags") or [])
            if str(t).strip()
        ],
        "subtopic_keywords": [
            str(k).lower().strip()
            for k in (raw.get("subtopic_keywords") or [])
            if str(k).strip()
        ],
        "topics": [
            str(t).strip()
            for t in (raw.get("topics") or [])
            if str(t).strip()
        ],
    }
    valid_topics = load_topic_catalog_set()
    normalized["topics"] = [topic for topic in normalized["topics"] if topic in valid_topics]
    if normalized["subject"] not in subjects:
        normalized["subject"] = None
    if normalized["question_type"] not in {"coding", "coding_analysis", "mcq", "mixed"}:
        normalized["question_type"] = None
    if normalized["difficulty"] not in {"basic", "medium", "advanced"}:
        normalized["difficulty"] = None
    if normalized["count"] is not None:
        try:
            normalized["count"] = int(normalized["count"])
        except (TypeError, ValueError):
            normalized["count"] = None
    return normalized


def parse_query_intent_llm(query: str, rule_intent: dict | None = None) -> dict:
    api_key = get_openrouter_api_key()
    if not api_key:
        return {}

    catalog = build_llm_intent_catalog()
    rule_hint = json.dumps(rule_intent or {}, default=str)

    system_prompt = (
        "You are a highly accurate search intent parser for the Topin Educational Question Bank, designed for educational institute employees to retrieve relevant educational content for review, assessment preparation, content validation, curriculum development, and other academic workflows. Your role is to understand natural language queries, accurately identify the user's search intent and applicable filters, and enable precise retrieval of questions and related educational data without generating answers yourself. "
        "Return ONLY valid JSON. "
        "Use ONLY values that exist in the provided catalog. Never invent topics, tags, or subjects. "
        "If a field is unclear, return null or an empty list.\n"
        "Schema: {\n"
        "  \"subject\": null | one of catalog.subjects,\n"
        "  \"question_type\": null | \"coding\" | \"coding_analysis\" | \"mcq\" | \"mixed\",\n"
        "  \"difficulty\": null | \"basic\" | \"medium\" | \"advanced\",\n"
        "  \"count\": null | integer,\n"
        "  \"fetch_all\": boolean,\n"
        "  \"tags\": [strings],\n"
        "  \"subtopic_keywords\": [strings],\n"
        "  \"topics\": [exact topic names from catalog.topics]\n"
        "}\n"
        f"Catalog:\n{catalog}"
    )

    user_prompt = (
        f"Parse this user query into JSON only: {query}\n"
        f"Rule-based parser hint (may be incomplete): {rule_hint}"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        raw = _extract_json(content)
        return _normalize_llm_intent(raw)
    except Exception:
        return {}


def merge_llm_intent(intent: dict, llm_intent: dict) -> dict:
    merged = dict(intent)
    if not merged.get("subject") and llm_intent.get("subject"):
        merged["subject"] = llm_intent["subject"]
    if not merged.get("question_type") and llm_intent.get("question_type"):
        merged["question_type"] = llm_intent["question_type"]
    if not merged.get("difficulty") and llm_intent.get("difficulty"):
        merged["difficulty"] = llm_intent["difficulty"]
    if not merged.get("has_explicit_count") and llm_intent.get("count") is not None:
        merged["limit"] = llm_intent["count"]
        merged["fetch_all"] = False
        merged["has_explicit_count"] = True
    if not merged.get("has_explicit_count") and llm_intent.get("fetch_all"):
        merged["limit"] = None
        merged["fetch_all"] = True
        merged["has_explicit_count"] = True
    if not merged.get("tags") and llm_intent.get("tags"):
        _, _, tag_to_questions, _ = load_question_tag_index()
        merged["tags"] = canonicalize_tags(llm_intent["tags"], tag_to_questions)
    elif llm_intent.get("tags"):
        _, _, tag_to_questions, _ = load_question_tag_index()
        merged_tags = list(dict.fromkeys((merged.get("tags") or []) + llm_intent["tags"]))
        merged["tags"] = canonicalize_tags(merged_tags, tag_to_questions)
    if llm_intent.get("topics") and not merged.get("topics"):
        merged["topics"] = llm_intent["topics"]
        merged["primary_topics"] = list(llm_intent["topics"])
    elif llm_intent.get("topics"):
        merged_topics = list(dict.fromkeys((merged.get("topics") or []) + llm_intent["topics"]))
        merged["topics"] = filter_topics_for_intent(merged_topics, merged)
    if not merged.get("subtopic_keywords") and llm_intent.get("subtopic_keywords") and not merged.get("tags"):
        merged["subtopic_keywords"] = llm_intent["subtopic_keywords"]
    return merged


def generate_search_intro_llm(
    user_query: str,
    intent: dict,
    result_count: int,
    collections: list[str],
    total_available: int | None = None,
    no_results: bool = False,
) -> str:
    """Friendly natural-language summary using only verified search facts."""
    fallback = describe_intent(intent, collections, result_count, total_available)
    api_key = get_openrouter_api_key()
    if not api_key:
        return fallback

    facts = {
        "user_query": user_query,
        "result_count": result_count,
        "collections": [name.removesuffix("_questions") for name in collections],
        "topics": intent.get("topics") or [],
        "tags": intent.get("tags") or [],
        "subject": intent.get("subject"),
        "question_type": intent.get("question_type"),
        "difficulty": intent.get("difficulty"),
        "total_available": total_available,
        "no_results": no_results,
    }

    system_prompt = (
        "You are the Topin question search assistant. Write a brief, friendly 1-2 sentence intro. "
        "Use ONLY the JSON facts provided. "
        "Do NOT invent, list, or describe specific questions. "
        "Do NOT change the result_count. "
        "If no_results is true, politely say nothing matched and suggest refining the query."
    )
    user_prompt = f"Facts JSON:\n{json.dumps(facts, indent=2)}"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
        return content or fallback
    except Exception:
        return fallback


def get_question_id(item: dict) -> str:
    parsed = parse_question_content(item["content"], item.get("metadata"))
    raw = _raw_value(parsed.get("question_id") or item.get("metadata", {}).get("question_id", ""))
    return normalize_question_id(raw) if raw else ""


def get_item_tag_tokens(item: dict, tag_index: dict[str, set[str]]) -> set[str]:
    qid = get_question_id(item)
    if qid and qid in tag_index:
        return tag_index[qid]

    tokens: set[str] = set()
    metadata = item.get("metadata") or {}
    for key in ("unit_tag", "module_tag", "course_tag", "grit_tag", "tags", "extra_tags"):
        raw = _raw_value(metadata.get(key, ""))
        if raw:
            tokens.add(raw.upper().replace(" ", "_"))
            for part in raw.split(","):
                cleaned = part.strip().upper().replace(" ", "_")
                if cleaned:
                    tokens.add(cleaned)

    parsed = parse_question_content(item["content"], metadata)
    for field in ("tags", "unit_tag", "module_tag", "course_tag", "grit_tag"):
        raw = _raw_value(parsed.get(field, ""))
        if raw:
            tokens.add(raw.upper().replace(" ", "_"))
            for part in raw.split(","):
                cleaned = part.strip().upper().replace(" ", "_")
                if cleaned:
                    tokens.add(cleaned)
    return tokens


def question_has_tags(question_id: str, required_tags: list[str], tag_index: dict[str, set[str]]) -> bool:
    if not required_tags:
        return True
    qid = normalize_question_id(question_id) if question_id else ""
    question_tags = tag_index.get(qid, set()) if qid else set()
    return all(normalize_tag_key(required) in {normalize_tag_key(t) for t in question_tags} for required in required_tags)


def question_item_has_tags(item: dict, required_tags: list[str], tag_index: dict[str, set[str]]) -> bool:
    if not required_tags:
        return True
    tokens = get_item_tag_tokens(item, tag_index)
    token_keys = {normalize_tag_key(t) for t in tokens}
    return all(normalize_tag_key(required) in token_keys for required in required_tags)


def filter_hits_by_tags(hits: list[dict], required_tags: list[str], tag_index: dict[str, set[str]]) -> list[dict]:
    if not required_tags:
        return hits
    return [hit for hit in hits if question_item_has_tags(hit, required_tags, tag_index)]


def parse_difficulty_from_query(query: str) -> str | None:
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    if re.search(r"\b(advanced|hard|difficult)\b", normalized):
        return "advanced"
    if re.search(r"\b(basic|easy|beginner)\b", normalized):
        return "basic"
    if re.search(r"\b(medium|intermediate|moderate)\b", normalized):
        return "medium"
    return None


def normalize_difficulty(value: str) -> str:
    cleaned = str(value).upper().replace("DIFFICULTY_", "").strip()
    return re.sub(r"\s+", " ", cleaned)


def get_item_difficulty(item: dict) -> str:
    metadata = item.get("metadata", {})
    if metadata.get("difficulty"):
        return normalize_difficulty(metadata["difficulty"])
    parsed = parse_question_content(item["content"], metadata)
    return normalize_difficulty(parsed.get("difficulty", ""))


def difficulty_matches(stored_difficulty: str, requested_level: str) -> bool:
    allowed = DIFFICULTY_GROUPS.get(requested_level, set())
    return normalize_difficulty(stored_difficulty) in allowed


def filter_hits_by_difficulty(hits: list[dict], requested_level: str | None) -> list[dict]:
    if not requested_level:
        return hits
    return [hit for hit in hits if difficulty_matches(get_item_difficulty(hit), requested_level)]


def summarize_difficulties(hits: list[dict]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for hit in hits:
        level = get_item_difficulty(hit) or "UNKNOWN"
        summary[level] = summary.get(level, 0) + 1
    return summary


def build_no_match_message(intent: dict, available_hits: list[dict]) -> str:
    parts = []
    if intent.get("subject"):
        parts.append(intent["subject"])
    if intent.get("question_type"):
        parts.append(intent["question_type"].replace("_", " "))
    subject_label = " ".join(parts) if parts else "selected"

    if intent.get("difficulty"):
        available = summarize_difficulties(available_hits)
        if not available:
            return f"No **{intent['difficulty']}** {subject_label} questions found in the database."
        breakdown = ", ".join(f"**{level.title()}** ({count})" for level, count in sorted(available.items()))
        return (
            f"No **{intent['difficulty']}** {subject_label} questions found in the database. "
            f"Available difficulties: {breakdown}."
        )

    if intent.get("tags"):
        return f"No questions found for {subject_label} with tags: {', '.join(intent['tags'])}."

    return f"No matching {subject_label} questions found in the database."


def apply_strict_filters(hits: list[dict], intent: dict, tag_index: dict[str, set[str]]) -> list[dict]:
    if is_tag_primary_intent(intent):
        scoped_intent = intent_without_topic_scope(intent)
        if intent.get("question_type") or intent.get("mixed") or intent.get("subject"):
            hits = filter_hits_by_intent_scope(hits, scoped_intent)
        hits = filter_hits_by_tags(hits, intent.get("tags") or [], tag_index)
        hits = filter_hits_by_difficulty(hits, intent.get("difficulty"))
        return hits

    hits = filter_hits_by_intent_scope(hits, intent)
    hits = filter_hits_by_tags(hits, intent.get("tags") or [], tag_index)
    hits = filter_hits_by_difficulty(hits, intent.get("difficulty"))
    if intent.get("subtopics"):
        hits = filter_hits_by_subtopics(
            hits,
            intent["subtopics"],
            exempt_collections=intent.get("primary_topics") or [],
        )
    return hits


def intent_has_search_scope(intent: dict) -> bool:
    return bool(
        intent.get("question_id")
        or intent.get("subject")
        or intent.get("topics")
        or intent.get("subtopic_keywords")
        or intent.get("tags")
    )


def intent_is_scoped(intent: dict) -> bool:
    return bool(
        intent.get("subject")
        or intent.get("topics")
        or intent.get("difficulty")
        or intent.get("tags")
        or intent.get("subtopics")
        or intent.get("question_type")
    )


def filter_hits_by_intent_scope(hits: list[dict], intent: dict) -> list[dict]:
    """Drop hits that do not belong to the requested subject/topic/type collections."""
    if not intent_has_filters(intent):
        return hits

    filtered: list[dict] = []
    for hit in hits:
        collection = hit.get("collection", "")
        if collection == "all_questions":
            continue
        if collection in {"unassigned_questions", "unassigned_mcq_questions", "unassigned_coding_questions"}:
            if is_tag_primary_intent(intent):
                filtered.append(hit)
            continue
        if not collection.endswith("_questions"):
            continue
        if not collection_matches_intent(collection, intent):
            continue
        if not hit_matches_search_intent(hit, intent):
            continue
        filtered.append(hit)
    return filtered


def finalize_search_results(
    hits: list[dict],
    intent: dict,
    tag_index: dict[str, set[str]],
    available_hits: list[dict] | None = None,
) -> list[dict]:
    """Validate every hit against intent filters; never return loosely matched results."""
    if not hits:
        return hits

    if needs_strict_filtering(intent) or intent_is_scoped(intent):
        hits = apply_strict_filters(hits, intent, tag_index)
    else:
        hits = filter_hits_by_intent_scope(hits, intent)

    return hits


def needs_strict_filtering(intent: dict) -> bool:
    return bool(intent.get("tags") or intent.get("difficulty") or intent.get("subtopics"))


def rerank_hits_by_similarity(hits: list[dict], query: str, embeddings, limit: int) -> list[dict]:
    if not hits:
        return hits

    query_vector = embeddings.embed_query(query)
    batch_size = 64
    scored: list[dict] = []

    for start in range(0, len(hits), batch_size):
        batch = hits[start : start + batch_size]
        vectors = embeddings.embed_documents([item["content"] for item in batch])
        for item, vector in zip(batch, vectors):
            score = sum(left * right for left, right in zip(query_vector, vector))
            scored.append({**item, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_qdrant_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=20)

    if not client.collection_exists(collection_name="all_questions"):
        client.create_collection(
            collection_name="all_questions",
            vectors_config=qdrant_models.VectorParams(
                size=384,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
    return client


@st.cache_resource
def load_vector_store():
    return QdrantVectorStore(
        client=load_qdrant_client(),
        collection_name="all_questions",
        embedding=load_embeddings(),
    )


@st.cache_data(ttl=300)
def get_searchable_collections(qdrant_url: str, api_key: str):
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=20)
    return [
        col.name
        for col in client.get_collections().collections
        if col.name.endswith("_questions") and col.name != "all_questions"
    ]


def parse_result_limit(query: str) -> tuple[int | None, bool]:
    """Return (limit, fetch_all). fetch_all=True means return every question in matched collections."""
    normalized = re.sub(r"\s+", " ", query.lower().strip())

    if re.search(
        r"\b(all|entire|full|every|complete|whole)\b.*\b(question|problem|mcq|item)s?\b",
        normalized,
    ):
        return None, True
    if is_comprehensive_content_query(query):
        return None, True
    if re.search(r"\b(show|give|list|fetch|get|display)\s+(me\s+)?(all|every|entire)\b", normalized):
        return None, True
    if re.search(r"\b(all|every)\s+(the\s+)?(question|problem|mcq)s?\b", normalized):
        return None, True

    match = re.search(r"\b(top|first|last)\s+(\d+)\b", normalized)
    if match:
        return min(int(match.group(2)), MAX_RESULT_LIMIT), False

    if re.search(r"\b(questions?|problems?|mcqs?|results?|items?)\b", normalized):
        match = re.search(r"\b(\d+)\b", normalized)
        if match:
            return min(int(match.group(1)), MAX_RESULT_LIMIT), False

    match = re.search(r"\b(give|show|get|list|fetch|display)\s+(me\s+)?(\d+)\b", normalized)
    if match:
        return min(int(match.group(3)), MAX_RESULT_LIMIT), False

    return DEFAULT_RESULT_LIMIT, False


def parse_count_info(query: str) -> dict:
    limit, fetch_all = parse_result_limit(query)
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    has_explicit_count = fetch_all or is_comprehensive_content_query(query) or bool(
        re.search(r"\b(give|show|get|list|fetch|display)\s+(me\s+)?(\d+|all|every)\b", normalized)
        or re.search(r"\b(top|first|last)\s+\d+\b", normalized)
        or (
            re.search(r"\b(questions?|problems?|mcqs?|results?|items?)\b", normalized)
            and re.search(r"\b\d+\b", normalized)
        )
    )
    return {
        "limit": None if fetch_all else limit,
        "fetch_all": fetch_all,
        "has_explicit_count": has_explicit_count,
    }


def parse_query_intent(query: str) -> dict:
    question_id = parse_question_id_from_query(query)
    if question_id:
        return build_question_id_intent(question_id)

    tags = parse_query_tags(query)
    keyword_query = strip_structured_tags(query) if has_curriculum_tags(tags) else query
    normalized = re.sub(r"\s+", " ", keyword_query.lower().strip())

    question_type = None
    if re.search(r"\bmixed\b", normalized):
        question_type = None
    elif re.search(r"coding\s*analysis|code\s*analysis|coding_analysis", normalized):
        question_type = "coding_analysis"
    elif re.search(r"\b(mcq|mcqs|multiple\s*choice)\b", normalized):
        question_type = "mcq"
    elif re.search(
        r"\b(coding|code\s*question|write\s+(a\s+)?(program|code|function))\b",
        normalized,
    ):
        question_type = "coding"

    subtopic = None
    for token, value in (("selenium", "selenium"), ("oops", "oops"), ("oop", "oops"), ("dsa", "dsa")):
        if re.search(rf"\b{token}\b", normalized):
            subtopic = value
            break

    subject = None
    for alias, normalized_subject in sorted(SUBJECT_ALIASES, key=lambda item: -len(item[0])):
        pattern = rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])"
        if re.search(pattern, normalized):
            subject = normalized_subject
            break

    count_info = parse_count_info(query)
    difficulty = parse_difficulty_from_query(keyword_query)
    primary_topics: list[str] = []

    if has_curriculum_tags(tags):
        _, _tag_collections, tags = resolve_tag_search(tags)
        topics = []
        primary_topics = []
        subtopics = []
        subtopic_keywords = []
        subject = None
        subtopic = None
        tag_primary = True
    else:
        tag_primary = False
        subtopic_info = resolve_subtopics_from_query(keyword_query, question_type)
        topic_info = resolve_topics_from_query(keyword_query, question_type)
        library_info = resolve_library_topics_from_query(keyword_query, question_type)

        keywords = list(subtopic_info.get("subtopic_keywords") or [])
        for keyword in topic_info.get("topic_keywords") or []:
            if keyword not in keywords:
                keywords.append(keyword)
        if library_info["library"] and library_info["library"] not in keywords:
            keywords.append(library_info["library"])

        topics_set = set(subtopic_info.get("topics") or [])
        topics_set.update(topic_info.get("topics") or [])
        topics_set.update(library_info.get("topics") or [])
        all_subtopics = sorted(subtopic_info.get("subtopics") or [])
        subtopic_keywords = keywords

        if keywords and topics_set:
            topics, primary_topics, subtopics = enrich_cross_collection_intent(
                keywords,
                sorted(topics_set),
                all_subtopics,
                subject,
                question_type,
                library_info,
            )
            if library_info["library"]:
                subtopic = subtopic or library_info["library"]
            elif topic_info.get("topic_keywords"):
                subtopic = subtopic or topic_info["topic_keywords"][0]
            elif subtopics and not subtopic:
                subtopic = subtopics[0].replace("SUB_TOPIC_", "").lower()
        elif keywords and not topics_set:
            topics = []
            primary_topics = []
            subtopics = sorted(
                value for value in all_subtopics if _subtopic_matches_keywords(value, keywords)
            )
            if subtopics and not subtopic:
                subtopic = subtopics[0].replace("SUB_TOPIC_", "").lower()
        else:
            topics = sorted(topics_set)
            subtopics = all_subtopics
            if subtopics and not subtopic:
                subtopic = subtopics[0].replace("SUB_TOPIC_", "").lower()

        if not subject:
            subject = subtopic_info.get("subject")
            if not subject and library_info["library"] in {"pandas", "numpy", "matplotlib"}:
                subject = "python"

        if subject and subject not in subtopic_keywords:
            if re.search(rf"\b{re.escape(subject)}\b", normalized):
                subtopic_keywords.append(subject)

    base_intent = {
        "subject": subject,
        "question_type": question_type,
        "subtopic": subtopic,
        "subtopics": subtopics,
        "subtopic_keywords": subtopic_keywords,
        "topics": topics,
        "primary_topics": primary_topics,
        "subject_inferred": bool(subtopics and not subject and topics),
        "tags": tags,
        "tag_primary": tag_primary,
        "difficulty": difficulty,
        "mixed": bool(re.search(r"\bmixed\b", normalized)),
        "limit": count_info["limit"],
        "fetch_all": count_info["fetch_all"],
        "has_explicit_count": count_info["has_explicit_count"],
        "has_explicit_difficulty": difficulty is not None,
    }

    if has_curriculum_tags(base_intent.get("tags") or []):
        return finalize_query_intent(base_intent, query)

    llm_intent = parse_query_intent_llm(query, base_intent)
    merged_intent = merge_llm_intent(base_intent, llm_intent)
    return finalize_query_intent(merged_intent, query)


def get_missing_selection_fields(intent: dict, query: str = "") -> list[str]:
    if intent.get("question_id"):
        return []

    missing: list[str] = []
    has_scope = bool(
        intent.get("topics")
        or intent.get("tags")
        or intent.get("subject")
        or intent.get("subtopic_keywords")
        or intent.get("subtopics")
    )

    if intent.get("tags"):
        return []

    if not intent.get("topics") and not has_scope:
        missing.append("topic")

    if not intent.get("subject") and not intent.get("topics") and not intent.get("subtopic_keywords"):
        missing.append("subject")

    has_question_type = bool(intent.get("question_type") or intent.get("mixed"))
    if not has_question_type and intent.get("topics"):
        has_question_type = bool(infer_question_type_from_topics(intent["topics"]))
    if not has_question_type and intent.get("topics"):
        if intent.get("difficulty") or intent.get("has_explicit_count"):
            has_question_type = True
    if not has_question_type and query:
        has_question_type = bool(infer_question_type_from_query(query))
    if not has_question_type:
        missing.append("question_type")

    if not intent.get("has_explicit_count") and not intent.get("fetch_all"):
        missing.append("count")

    if not intent.get("difficulty") and not intent.get("has_explicit_difficulty"):
        missing.append("difficulty")

    return missing


def build_selection_message(missing: list[str]) -> str:
    labels = {
        "subject": "**subject**",
        "topic": "**topic / collection**",
        "question_type": "**question type**",
        "count": "**how many questions**",
        "difficulty": "**difficulty level**",
    }
    items = [labels[field] for field in missing if field in labels]
    if not items:
        return "Please complete the remaining filters."
    if len(items) == 1:
        return f"Please select {items[0]}."
    if len(items) == 2:
        return f"Please select {items[0]} and {items[1]}."
    return f"Please select {', '.join(items[:-1])}, and {items[-1]}."


def requires_subject_type_selection(query: str, intent: dict) -> tuple[bool, str]:
    intent = finalize_query_intent(intent, query)
    missing = get_missing_selection_fields(intent, query)
    if not missing:
        return False, ""

    normalized = query.lower()
    is_question_request = bool(
        re.search(
            r"\b(give|show|get|list|fetch|display|find|want|need|questions?|mcqs?|problems?)\b",
            normalized,
        )
    )
    has_partial = bool(
        intent.get("subject")
        or intent.get("question_type")
        or intent.get("mixed")
        or intent.get("has_explicit_count")
        or intent.get("has_explicit_difficulty")
        or intent.get("tags")
        or intent.get("subtopics")
        or intent.get("topics")
    )

    if not is_question_request and not has_partial:
        return False, ""

    return True, build_selection_message(missing)


def build_intent_from_selection(partial_intent: dict, selections: dict) -> dict:
    intent = dict(partial_intent)

    subject = selections.get("subject") or intent.get("subject")
    if subject:
        intent["subject"] = subject

    question_type = selections.get("question_type")
    if question_type:
        if question_type == "mixed":
            intent["question_type"] = None
            intent["mixed"] = True
        else:
            intent["question_type"] = question_type
            intent["mixed"] = False

    count_choice = selections.get("count_choice")
    if count_choice:
        if count_choice == "all":
            intent["fetch_all"] = True
            intent["limit"] = None
        else:
            intent["fetch_all"] = False
            intent["limit"] = int(count_choice)
        intent["has_explicit_count"] = True

    difficulty = selections.get("difficulty")
    if difficulty:
        intent["difficulty"] = None if difficulty == "any" else difficulty
        intent["has_explicit_difficulty"] = True

    topic = selections.get("topic")
    if topic:
        intent["topics"] = [topic]
        intent["primary_topics"] = [topic]

    return intent


def build_query_from_intent(intent: dict) -> str:
    parts = []
    if intent.get("fetch_all"):
        parts.append("all")
    elif intent.get("limit"):
        parts.append(str(intent["limit"]))
    if intent.get("subject"):
        parts.append(intent["subject"])
    if intent.get("mixed"):
        parts.append("mixed")
    elif intent.get("question_type"):
        parts.append(intent["question_type"].replace("_", " "))
    if intent.get("subtopic"):
        parts.append(intent["subtopic"])
    elif intent.get("subtopic_keywords"):
        parts.extend(intent["subtopic_keywords"])
    if intent.get("difficulty"):
        parts.append(intent["difficulty"])
    if intent.get("tags"):
        parts.extend(intent["tags"])
    parts.append("questions")
    return " ".join(parts)


def _subject_matches_collection(base: str, subject: str) -> bool:
    if subject == "c":
        return bool(re.match(r"topic_c_(?!sharp)", base))

    if subject == "python" and base in STANDALONE_TOPIC_TYPES:
        return True

    prefixes = SUBJECT_COLLECTION_PREFIXES.get(subject, [f"topic_{subject}_"])
    return any(base.startswith(prefix) for prefix in prefixes)


def collection_matches_intent(collection_name: str, intent: dict) -> bool:
    base = collection_name.removesuffix("_questions")
    subject = intent.get("subject")
    question_type = intent.get("question_type")
    subtopic = intent.get("subtopic")
    topics = intent.get("topics") or []
    mixed = intent.get("mixed")

    if not mixed and question_type:
        coll_type = collection_question_type(base)
        if coll_type:
            if coll_type != question_type:
                return False
        elif question_type == "coding_analysis":
            if "_coding_analysis" not in base:
                return False
        elif question_type == "coding":
            if "_coding_analysis" in base or "_coding" not in base:
                return False
        elif question_type == "mcq" and "_mcq" not in base:
            return False

    if topics and not intent.get("tag_primary"):
        if base not in topics:
            return False
    elif subtopic and f"_{subtopic}_" not in base:
        return False

    if subject and not topics and not intent.get("tag_primary") and not _subject_matches_collection(base, subject):
        return False

    return True


def intent_has_filters(intent: dict) -> bool:
    if intent.get("mixed") and intent.get("subject"):
        return True
    return any(
        intent.get(key)
        for key in ("subject", "question_type", "subtopic", "subtopics", "topics", "tags", "difficulty")
    )


def filter_collections(collections: list[str], intent: dict) -> list[str]:
    if not intent_has_filters(intent):
        return collections
    return [name for name in collections if collection_matches_intent(name, intent)]


def describe_intent(intent: dict, collections: list[str], result_count: int, total_available: int | None = None) -> str:
    parts = []
    if intent.get("subject"):
        parts.append(intent["subject"])
    if intent.get("subtopic"):
        parts.append(intent["subtopic"])
    elif intent.get("subtopic_keywords"):
        parts.append(" / ".join(intent["subtopic_keywords"]))
    if intent.get("question_type"):
        parts.append(intent["question_type"].replace("_", " "))
    elif intent.get("mixed"):
        parts.append("mixed")
    if intent.get("difficulty"):
        parts.append(intent["difficulty"])
    if intent.get("tags"):
        parts.append("tags: " + ", ".join(intent["tags"]))
    label = " ".join(parts) if parts else "all topics"
    summary = f"Showing **{result_count}** question(s) from **{label}** ({len(collections)} collection(s))."
    if total_available is not None and result_count < total_available:
        summary += f" ({total_available - result_count} more available — ask for a higher number or `all questions`.)"
    return summary


def _search_collections(client, vector, collections, per_collection_limit):
    hits = []

    def search_one(collection_name):
        results = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=per_collection_limit,
            with_payload=True,
        )
        return [
            {
                "score": point.score,
                "content": point.payload.get("page_content", ""),
                "collection": collection_name,
                "metadata": point.payload.get("metadata", {}) or {},
            }
            for point in results.points
        ]

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(search_one, name) for name in collections]
        for future in as_completed(futures):
            hits.extend(future.result())

    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits


def _count_available_points(client, collections: list[str]) -> int:
    return sum(client.get_collection(name).points_count for name in collections)


def _fetch_all_from_collections(client, collections: list[str], max_points: int = MAX_RESULT_LIMIT):
    hits = []
    seen_ids: set = set()

    for collection_name in collections:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                if point.id in seen_ids:
                    continue
                seen_ids.add(point.id)
                hits.append(
                    {
                        "score": 1.0,
                        "content": point.payload.get("page_content", ""),
                        "collection": collection_name,
                        "metadata": point.payload.get("metadata", {}) or {},
                    }
                )
                if len(hits) >= max_points:
                    return hits[:max_points]

            if offset is None:
                break

    return hits[:max_points]


def _fetch_complete_collections(client, collections: list[str]) -> list[dict]:
    """Scroll each collection fully — tag searches must not share a global point cap."""
    hits: list[dict] = []
    seen_ids: set = set()

    for collection_name in collections:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                if point.id in seen_ids:
                    continue
                seen_ids.add(point.id)
                hits.append(
                    {
                        "score": 1.0,
                        "content": point.payload.get("page_content", ""),
                        "collection": collection_name,
                        "metadata": point.payload.get("metadata", {}) or {},
                    }
                )

            if offset is None:
                break

    return hits


def _fetch_hits_for_collections(
    client,
    collections: list[str],
    *,
    required_tags: list[str] | None = None,
    matched_ids: set[str] | None = None,
    tag_primary: bool = False,
    total_available: int | None = None,
) -> list[dict]:
    if tag_primary and matched_ids:
        tag_signature = "|".join(sorted(required_tags or []))
        return fetch_cached_tag_primary_hits(tag_signature, tuple(sorted(matched_ids)))
    if required_tags:
        return _fetch_complete_collections(client, collections)
    cap = min(total_available or MAX_RESULT_LIMIT, ALL_FETCH_CAP)
    return _fetch_all_from_collections(client, collections, max_points=cap)


def search_all_collections(client, embeddings, query, intent_override: dict | None = None):
    intent = dict(intent_override) if intent_override else parse_query_intent(query)
    intent = sanitize_intent_topics(intent)

    question_id = intent.get("question_id")
    if question_id:
        _, _, _, question_topics = load_question_tag_index()
        if question_id not in question_topics:
            return [], (
                f"No question found for ID **{question_id}** in the Topin database."
            )
        hit = fetch_question_hit_by_id(client, question_id, intent.get("topics"))
        if not hit:
            topic = question_topics[question_id]
            return [], (
                f"Question ID **{question_id}** exists in the index (`{topic}`) "
                "but was not found in the vector database."
            )
        collection = hit.get("collection", "").removesuffix("_questions")
        label = f"Showing question **{question_id}** from `{collection}`."
        return [hit], label

    if intent.get("has_explicit_count"):
        limit = intent.get("limit")
        fetch_all = bool(intent.get("fetch_all"))
    else:
        limit, fetch_all = parse_result_limit(query)

    tag_index, _tag_display, _tag_to_questions, _question_topics = load_question_tag_index()
    matched_ids: set[str] = set()
    required_tags = intent.get("tags") or []
    tag_primary = is_tag_primary_intent(intent)
    if required_tags:
        matched_ids, tag_collections, canonical_tags = resolve_tag_search(required_tags)
        required_tags = canonical_tags
        intent = dict(intent)
        intent["tags"] = canonical_tags
        if not matched_ids:
            return [], (
                f"No questions found for tag(s): **{', '.join(canonical_tags)}**. "
                "Check the unit/module/course tag spelling in your query."
            )
    else:
        tag_collections = []

    requested_difficulty = intent.get("difficulty")

    if required_tags and intent_has_filters(intent) and not intent.get("has_explicit_count"):
        fetch_all = True
    if tag_primary and required_tags:
        fetch_all = True
    if is_comprehensive_content_query(query) and intent_has_search_scope(intent):
        fetch_all = True
    if requested_difficulty and intent_has_search_scope(intent) and not intent.get("has_explicit_count"):
        fetch_all = True
    if intent.get("has_explicit_count") and intent.get("limit") == 1:
        fetch_all = False

    all_collections = get_searchable_collections(QDRANT_URL, QDRANT_API_KEY)
    if tag_primary and required_tags:
        collections_to_search = [name for name in tag_collections if name in all_collections]
    else:
        targeted_collections = filter_collections(all_collections, intent)
        collections_to_search = targeted_collections if intent_has_filters(intent) else all_collections

        if tag_collections:
            collections_to_search = [
                name for name in tag_collections if name in all_collections
            ]
            if intent.get("question_type") or intent.get("mixed"):
                collections_to_search = filter_collections(collections_to_search, intent)
            if collections_to_search:
                intent = dict(intent)
                intent["topics"] = sorted(
                    name.removesuffix("_questions") for name in collections_to_search
                )
        elif required_tags and not collections_to_search:
            return [], build_no_match_message(intent, [])

    if intent_has_filters(intent) and not collections_to_search and not (tag_primary and matched_ids):
        return [], build_no_match_message(intent, [])

    total_available = _count_available_points(client, collections_to_search) if collections_to_search else 0
    actual_limit = limit or DEFAULT_RESULT_LIMIT

    if needs_strict_filtering(intent):
        if not intent_has_search_scope(intent):
            return [], (
                "To filter by difficulty or tags, include a topic or subject in your query — "
                "e.g. `10 easy terraform mcqs` or `advanced git questions`."
            )
        cap = min(total_available, ALL_FETCH_CAP)
        available_hits = _fetch_hits_for_collections(
            client,
            collections_to_search,
            required_tags=required_tags,
            matched_ids=matched_ids,
            tag_primary=tag_primary,
            total_available=total_available,
        )
        hits = apply_strict_filters(available_hits, intent, tag_index)
        if not hits:
            return [], build_no_match_message(intent, available_hits)
        if not fetch_all:
            hits = rerank_hits_by_similarity(hits, query, embeddings, actual_limit)
        total_for_label = len(matched_ids) if tag_primary and matched_ids else len(available_hits)
        label = describe_intent(intent, collections_to_search, len(hits), total_for_label)
        return hits, label

    if fetch_all:
        if not intent_has_filters(intent):
            return [], (
                "To list all questions, please specify a subject and type — "
                "e.g. `show all python coding questions`."
            )
        cap = min(total_available, ALL_FETCH_CAP)
        results = _fetch_hits_for_collections(
            client,
            collections_to_search,
            required_tags=required_tags,
            matched_ids=matched_ids,
            tag_primary=tag_primary,
            total_available=total_available,
        )
        results = finalize_search_results(results, intent, tag_index, results)
        if not results:
            return [], build_no_match_message(intent, [])
        total_for_label = len(matched_ids) if tag_primary and matched_ids else total_available
        label = describe_intent(intent, collections_to_search, len(results), total_for_label)
        return results, label

    vector = embeddings.embed_query(query)

    if intent_has_filters(intent):
        if len(collections_to_search) == 1:
            results = client.query_points(
                collection_name=collections_to_search[0],
                query=vector,
                limit=min(actual_limit, MAX_RESULT_LIMIT),
                with_payload=True,
            )
            hits = [
                {
                    "score": point.score,
                    "content": point.payload.get("page_content", ""),
                    "collection": collections_to_search[0],
                    "metadata": point.payload.get("metadata", {}) or {},
                }
                for point in results.points
            ]
        else:
            per_collection = max(3, actual_limit // max(len(collections_to_search), 1) + 1)
            hits = _search_collections(client, vector, collections_to_search, per_collection)
            hits = hits[:actual_limit]

        hits = finalize_search_results(hits, intent, tag_index)
        if not hits:
            return [], build_no_match_message(intent, [])

        label = describe_intent(intent, collections_to_search, len(hits), total_available)
        return hits, label

    if not intent_is_scoped(intent) and client.get_collection("all_questions").points_count > 0:
        db = load_vector_store()
        docs = db.similarity_search(query, k=min(actual_limit, MAX_RESULT_LIMIT))
        hits = [
            {
                "score": 1.0,
                "content": doc.page_content,
                "collection": "all_questions",
                "metadata": doc.metadata or {},
            }
            for doc in docs
        ]
        return hits, describe_intent(intent, collections_to_search, len(hits))

    per_collection_limit = max(2, actual_limit // 20 + 1)
    hits = _search_collections(client, vector, collections_to_search, per_collection_limit)
    hits = hits[:actual_limit]
    return hits, describe_intent(intent, collections_to_search, len(hits))


def fetch_pool_for_intent(client, embeddings, intent: dict, query: str) -> list[dict]:
    """Return all questions matching intent (used for 'give me more' follow-ups)."""
    tag_index, _, _tag_to_questions, _question_topics = load_question_tag_index()
    all_collections = get_searchable_collections(QDRANT_URL, QDRANT_API_KEY)
    required_tags = intent.get("tags") or []
    matched_ids: set[str] = set()
    tag_collections: list[str] = []
    tag_primary = is_tag_primary_intent(intent)
    if required_tags:
        matched_ids, tag_collections, canonical_tags = resolve_tag_search(required_tags)
        intent = dict(intent)
        intent["tags"] = canonical_tags

    if tag_primary and required_tags:
        collections_to_search = [name for name in tag_collections if name in all_collections]
    else:
        collections_to_search = filter_collections(all_collections, intent)
        if tag_collections:
            collections_to_search = [
                name for name in tag_collections if name in all_collections
            ]
            if intent.get("question_type") or intent.get("mixed"):
                collections_to_search = filter_collections(collections_to_search, intent)
    if not collections_to_search and not (tag_primary and matched_ids):
        return []

    total_available = _count_available_points(client, collections_to_search) if collections_to_search else 0
    cap = min(total_available, ALL_FETCH_CAP)
    available_hits = _fetch_hits_for_collections(
        client,
        collections_to_search,
        required_tags=required_tags,
        matched_ids=matched_ids,
        tag_primary=tag_primary,
        total_available=total_available,
    )

    if needs_strict_filtering(intent):
        hits = apply_strict_filters(available_hits, intent, tag_index)
        if hits:
            if tag_primary:
                return hits
            return rerank_hits_by_similarity(hits, query, embeddings, len(hits))
        return []

    if intent_has_filters(intent):
        vector = embeddings.embed_query(query)
        if len(collections_to_search) == 1:
            results = client.query_points(
                collection_name=collections_to_search[0],
                query=vector,
                limit=min(cap, MAX_RESULT_LIMIT),
                with_payload=True,
            )
            hits = [
                {
                    "score": point.score,
                    "content": point.payload.get("page_content", ""),
                    "collection": collections_to_search[0],
                    "metadata": point.payload.get("metadata", {}) or {},
                }
                for point in results.points
            ]
        else:
            per_collection = max(5, cap // max(len(collections_to_search), 1))
            hits = _search_collections(client, vector, collections_to_search, per_collection)
        return finalize_search_results(hits, intent, tag_index, available_hits)

    return finalize_search_results(available_hits, intent, tag_index, available_hits)


def update_search_context(intent: dict, results: list[dict], result_pool: list[dict], query: str) -> None:
    st.session_state.search_context = {
        "intent": dict(intent),
        "results": results,
        "result_pool": result_pool,
        "shown_ids": {get_question_id(item) for item in results if get_question_id(item)},
        "query": query,
    }


def extract_subtopic_from_query(query: str) -> str | None:
    match = re.search(r"\b(SUB_TOPIC_[A-Z0-9_]+)\b", query, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    normalized = re.sub(r"\s+", " ", query.lower().strip())
    match = re.search(r"\bsubtopic\s+([a-z0-9_]+)\b", normalized)
    if match:
        token = match.group(1).upper()
        return token if token.startswith("SUB_TOPIC_") else f"SUB_TOPIC_{token}"

    match = re.search(r"\bonly\s+(?:give\s+)?([a-z0-9_]+)\s+subtopic", normalized)
    if match:
        token = match.group(1).upper()
        return token if token.startswith("SUB_TOPIC_") else f"SUB_TOPIC_{token}"

    return None


def get_item_subtopic(item: dict) -> str:
    parsed = parse_question_content(item["content"], item.get("metadata"))
    metadata = item.get("metadata") or {}
    return (
        _raw_value(parsed.get("subtopic", ""))
        or _raw_value(metadata.get("subtopic", ""))
        or ""
    ).upper()


def get_item_topic(item: dict) -> str:
    parsed = parse_question_content(item["content"], item.get("metadata"))
    metadata = item.get("metadata") or {}
    return _raw_value(parsed.get("topic") or metadata.get("topic", "")) or ""


def hit_matches_search_intent(hit: dict, intent: dict) -> bool:
    """
    Each question must prove it belongs to the requested scope.
    Dedicated collections are trusted; shared collections require exact subtopic/keyword proof.
    """
    requested_id = normalize_question_id(intent.get("question_id", ""))
    if requested_id:
        return get_question_id(hit) == requested_id

    if intent.get("tags"):
        return True

    collection = hit.get("collection", "").removesuffix("_questions")
    trusted = get_trusted_collections(intent)
    if collection in trusted:
        return True

    subtopic = get_item_subtopic(hit)
    subtopic_filters = [value.upper() for value in (intent.get("subtopics") or [])]
    keywords = intent.get("subtopic_keywords") or []

    if subtopic_filters:
        if subtopic in subtopic_filters:
            return True

    if keywords and subtopic and _subtopic_matches_keywords(subtopic, keywords):
        return True

    topic = get_item_topic(hit)
    if keywords and topic and any(_topic_name_matches_keyword(topic, keyword) for keyword in keywords):
        return True
    if keywords and any(_topic_name_matches_keyword(collection, keyword) for keyword in keywords):
        return True

    return False


def filter_hits_by_subtopics(
    hits: list[dict],
    subtopic_filters: list[str],
    exempt_collections: list[str] | None = None,
) -> list[dict]:
    if not subtopic_filters:
        return hits

    needles = [value.upper().strip() for value in subtopic_filters]
    exempt = {value.removesuffix("_questions") for value in (exempt_collections or [])}
    filtered = []
    for item in hits:
        collection = item.get("collection", "").removesuffix("_questions")
        if collection in exempt:
            filtered.append(item)
            continue
        subtopic = get_item_subtopic(item)
        if subtopic in needles:
            filtered.append(item)
    return filtered


def filter_results_by_subtopic(results: list[dict], subtopic: str) -> list[dict]:
    needle = subtopic.upper().strip()
    return [item for item in results if get_item_subtopic(item) == needle]


def is_context_refinement_query(query: str, context: dict | None) -> bool:
    if not context:
        return False
    if parse_follow_up(query, has_context=True):
        return True
    subtopic = extract_subtopic_from_query(query)
    return bool(
        subtopic
        and re.search(r"\b(only|filter|subtopic|just|remove|delete|more)\b", query, re.IGNORECASE)
    )


def restore_search_context_from_messages() -> None:
    if st.session_state.get("search_context"):
        return
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant" and message.get("results"):
            results = message["results"]
            st.session_state.search_context = {
                "intent": {},
                "results": results,
                "result_pool": results,
                "shown_ids": {get_question_id(item) for item in results if get_question_id(item)},
                "query": message.get("search_label", ""),
            }
            break


def parse_follow_up(query: str, has_context: bool = True) -> dict | None:
    normalized = re.sub(r"\s+", " ", query.lower().strip())

    if re.search(r"\b(remove|delete|drop|exclude)\b", normalized):
        indices = {int(value) for value in re.findall(r"\b(\d+)\b", normalized)}
        if indices:
            return {"type": "remove", "indices": indices}

    more_match = re.search(r"\b(?:give|show|get|add)?\s*(\d+)\s+more\b", normalized)
    if more_match:
        return {"type": "more", "count": int(more_match.group(1))}
    if re.search(r"\bmore\b", normalized) and re.search(
        r"\b(same|topic|existing|these|current|above)\b", normalized
    ):
        count_match = re.search(r"\b(\d+)\b", normalized)
        return {"type": "more", "count": int(count_match.group(1)) if count_match else 3}

    if re.search(r"\b(existing|current|these|above|shown|responded)\b", normalized) or re.search(
        r"\bfrom\s+(existing|current|these|above|shown|responded)\b", normalized
    ):
        keyword = "list"
        if "list" in normalized:
            keyword = "list"
        else:
            keyword_match = re.search(r"\bonly\s+([a-z0-9_ +]+?)\s+questions?\b", normalized)
            if keyword_match:
                keyword = keyword_match.group(1).strip()
        return {"type": "filter_existing", "keyword": keyword}

    if re.search(r"\bonly\s+list\b", normalized):
        return {"type": "filter_existing", "keyword": "list"}

    if has_context:
        subtopic = extract_subtopic_from_query(query)
        if subtopic and re.search(r"\b(only|filter|just|subtopic)\b", normalized):
            use_pool = not re.search(
                r"\b(existing|current|these|above|shown|responded)\b", normalized
            )
            return {"type": "filter_subtopic", "subtopic": subtopic, "use_pool": use_pool}

    return None


def filter_results_by_keyword(results: list[dict], keyword: str) -> list[dict]:
    needle = keyword.lower().strip()
    filtered = []
    for item in results:
        parsed = parse_question_content(item["content"], item.get("metadata"))
        haystack = " ".join(
            [
                parsed.get("question_text", ""),
                parsed.get("subtopic", ""),
                parsed.get("short_description", ""),
            ]
        ).lower()
        if needle in haystack:
            filtered.append(item)
    return filtered


def handle_follow_up(query: str, context: dict) -> tuple[list[dict], str] | None:
    action = parse_follow_up(query, has_context=True)
    if not action:
        return None

    if action["type"] == "remove":
        current = context["results"]
        kept = [item for index, item in enumerate(current, start=1) if index not in action["indices"]]
        removed = len(current) - len(kept)
        context["results"] = kept
        if not kept:
            return kept, "No questions left in your list after removal."
        return kept, f"Removed **{removed}** question(s). Showing **{len(kept)}** in your list."

    if action["type"] == "filter_existing":
        filtered = filter_results_by_keyword(context["results"], action["keyword"])
        context["results"] = filtered
        if not filtered:
            return filtered, f"No questions in your current list matched **{action['keyword']}**."
        return (
            filtered,
            f"Filtered your current list to **{len(filtered)}** question(s) matching **{action['keyword']}**.",
        )

    if action["type"] == "filter_subtopic":
        source = (
            context["results"]
            if action.get("use_pool") is False
            else (context.get("result_pool") or context["results"])
        )
        filtered = filter_results_by_subtopic(source, action["subtopic"])
        context["results"] = filtered
        context["shown_ids"] = {get_question_id(item) for item in filtered if get_question_id(item)}
        if not filtered:
            return (
                filtered,
                f"No questions matched subtopic **{action['subtopic']}** in your current search.",
            )
        scope = "current list" if action.get("use_pool") is False else "this topic"
        return (
            filtered,
            f"Showing **{len(filtered)}** question(s) with subtopic **{action['subtopic']}** from {scope}.",
        )

    if action["type"] == "more":
        pool = context.get("result_pool") or []
        shown_ids = context.get("shown_ids") or set()
        remaining = [item for item in pool if get_question_id(item) not in shown_ids]
        more = remaining[: action["count"]]
        if not more:
            return context["results"], "No more questions available for this topic."

        context["shown_ids"].update(get_question_id(item) for item in more if get_question_id(item))
        context["results"] = context["results"] + more
        return (
            context["results"],
            f"Added **{len(more)}** more question(s) from the same topic. Showing **{len(context['results'])}** total.",
        )

    return None


def append_assistant_results(
    results: list[dict],
    search_label: str,
    intent: dict,
    result_pool: list[dict] | None,
    query: str,
) -> None:
    if result_pool is not None:
        update_search_context(intent, results, result_pool, query)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": search_label,
            "search_label": search_label,
            "results": results,
            "matched_tags": intent.get("tags"),
        }
    )


def parse_question_content(page_content: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    fields = {"question_id": metadata.get("question_id", "")}
    current_key = None
    buffer: list[str] = []

    def flush():
        if current_key is not None:
            fields[current_key] = "\n".join(buffer).strip()

    for line in page_content.splitlines():
        matched = False
        for prefix, key in FIELD_PREFIXES:
            if line.startswith(prefix):
                flush()
                current_key = key
                value = line[len(prefix) :].strip()
                buffer = [value] if value else []
                matched = True
                break
        if not matched and current_key is not None:
            buffer.append(line)

    flush()
    return fields


def _clean_label(value: str) -> str:
    if not value or value.lower() in {"nan", "none", ""}:
        return ""
    return value.replace("DIFFICULTY_", "").replace("_", " ").strip().title()


def is_coding_question(item: dict) -> bool:
    collection = item.get("collection", "")
    if "coding_analysis" in collection:
        return False
    return "_coding" in collection


def html_to_markdown(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<hr\s*/?>", "\n\n---\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<i>(.*?)</i>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<pre>(.*?)</pre>", r"```\n\1\n```", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def normalize_markdown_question(text: str) -> str:
    """Render-friendly markdown for coding questions."""
    if not text:
        return ""
    text = html_to_markdown(text.replace("\r\n", "\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def plain_question_text(text: str) -> str:
    """Plain text for CSV export."""
    text = normalize_markdown_question(text)
    text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def get_tags_text(question_id: str, tag_display: dict[str, dict[str, str]]) -> str:
    return tag_display.get(question_id, {}).get("all_tags", "")


def render_question_tags(
    question_id: str,
    tag_index: dict[str, set[str]],
    matched_tags: list[str] | None = None,
) -> None:
    tokens = sorted(tag_index.get(question_id, set()))
    if not tokens:
        return

    matched = {tag.upper() for tag in (matched_tags or [])}
    rendered = []
    for token in tokens:
        if token in matched:
            rendered.append(f"**{token}**")
        else:
            rendered.append(f"`{token}`")
    st.markdown("**Tags:** " + " · ".join(rendered))


def result_to_row(item: dict, tag_display: dict[str, dict[str, str]]) -> dict:
    parsed = parse_question_content(item["content"], item.get("metadata"))
    question_id = _raw_value(parsed.get("question_id") or item.get("metadata", {}).get("question_id", ""))
    return {
        "question_id": question_id,
        "question": plain_question_text(parsed.get("question_text") or item["content"]),
        "topic": _raw_value(parsed.get("topic") or item["collection"].removesuffix("_questions")),
        "subtopic": _raw_value(parsed.get("subtopic", "")),
        "tags": get_tags_text(question_id, tag_display),
    }


def results_to_csv(results: list[dict], tag_display: dict[str, dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["question_id", "question", "topic", "subtopic", "tags"],
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for item in results:
        writer.writerow(result_to_row(item, tag_display))
    return buffer.getvalue()


def render_question_body(question_text: str, is_coding: bool) -> None:
    body = normalize_markdown_question(question_text)
    if is_coding:
        st.markdown("#### Problem")
    st.markdown(body, unsafe_allow_html=False)


def format_options(options_str: str) -> list[tuple[str, bool]]:
    if not options_str or options_str.lower() in {"nan", "none", ""}:
        return []
    try:
        options = json.loads(options_str)
    except json.JSONDecodeError:
        return []

    formatted = []
    for index, option in enumerate(options, start=1):
        letter = chr(64 + index) if index <= 26 else str(index)
        text = option.get("option_content", "")
        is_correct = bool(option.get("is_correct_option"))
        formatted.append((f"**{letter}.** {text}", is_correct))
    return formatted


def render_results(
    results: list[dict],
    search_label: str,
    message_key: str,
    matched_tags: list[str] | None = None,
) -> str:
    tag_index, tag_display, _, _ = load_question_tag_index()
    st.markdown(search_label)

    st.download_button(
        label=f"Download {len(results)} question(s) as CSV",
        data=results_to_csv(results, tag_display),
        file_name="topin_questions.csv",
        mime="text/csv",
        key=f"download_{message_key}",
    )

    if len(results) > RESULTS_PER_PAGE:
        total_pages = (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"page_{message_key}",
        )
        start = (page - 1) * RESULTS_PER_PAGE
        visible_results = results[start : start + RESULTS_PER_PAGE]
        st.caption(f"Page {page} of {total_pages}")
    else:
        start = 0
        visible_results = results

    text_parts = [search_label]
    for offset_idx, item in enumerate(visible_results):
        display_idx = start + offset_idx + 1
        parsed = parse_question_content(item["content"], item.get("metadata"))
        raw_topic = _raw_value(parsed.get("topic") or item["collection"].removesuffix("_questions"))
        raw_subtopic = _raw_value(parsed.get("subtopic", ""))
        topic = _clean_label(raw_topic) or raw_topic
        subtopic = _clean_label(raw_subtopic) or raw_subtopic
        difficulty = _clean_label(parsed.get("difficulty", item.get("metadata", {}).get("difficulty", "")))
        question_text = parsed.get("question_text") or item["content"]
        short_desc = _clean_label(parsed.get("short_description", ""))
        options = format_options(parsed.get("options", ""))
        coding = is_coding_question(item)

        header_parts = [f"Question {display_idx}"]
        if coding:
            header_parts.append("Coding")
        if difficulty:
            header_parts.append(difficulty)
        if subtopic:
            header_parts.append(subtopic)
        header = " · ".join(header_parts)

        with st.container(border=True):
            st.markdown(f"### {header}")
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Topic:** `{raw_topic}`")
                if raw_subtopic:
                    st.markdown(f"**Subtopic:** `{raw_subtopic}`")
            with cols[1]:
                if item["score"] < 0.99:
                    st.caption(f"Match: {item['score']:.0%}")

            qid = parsed.get("question_id") or item.get("metadata", {}).get("question_id")
            if qid:
                render_question_tags(qid, tag_index, matched_tags)

            if short_desc and not coding:
                st.info(short_desc)

            render_question_body(question_text, coding)

            if options:
                st.markdown("**Options**")
                for option_text, is_correct in options:
                    if is_correct:
                        st.success(option_text)
                    else:
                        st.markdown(option_text)

            if qid:
                st.caption(f"ID: `{qid}`")

        text_parts.append(f"### {header}\n**Topic:** {raw_topic}\n\n{normalize_markdown_question(question_text)}")

    return "\n\n".join(text_parts)


def render_selection_prompt(
    prompt_message: str,
    partial_intent: dict,
    original_query: str,
    message_key: str,
) -> None:
    st.markdown(prompt_message)

    missing = get_missing_selection_fields(partial_intent, original_query)
    detected = []
    if partial_intent.get("subject"):
        detected.append(f"Subject: **{partial_intent['subject']}**")
    if partial_intent.get("mixed"):
        detected.append("Type: **Mixed**")
    elif partial_intent.get("question_type"):
        detected.append(f"Type: **{partial_intent['question_type'].replace('_', ' ')}**")
    if partial_intent.get("has_explicit_count"):
        if partial_intent.get("fetch_all"):
            detected.append("Count: **All**")
        elif partial_intent.get("limit") is not None:
            detected.append(f"Count: **{partial_intent['limit']}**")
    if partial_intent.get("difficulty"):
        detected.append(f"Difficulty: **{partial_intent['difficulty']}**")
    if partial_intent.get("tags"):
        detected.append(f"Tags: **{', '.join(partial_intent['tags'])}**")
    if partial_intent.get("topics") and "topic" not in missing:
        detected.append(f"Topics: **{', '.join(partial_intent['topics'][:3])}**")
    if detected:
        st.caption("From your message: " + " · ".join(detected))

    selections: dict[str, str] = {}

    if "topic" in missing:
        topic_values = load_topic_catalog()
        topic_labels = {
            topic: topic.replace("topic_", "").replace("_", " ").title()
            for topic in topic_values
        }
        selected_topic = st.selectbox(
            "Topic / collection",
            ["— Select topic —"] + [topic_labels[topic] for topic in topic_values],
            key=f"topic_{message_key}",
        )
        if not selected_topic.startswith("—"):
            selections["topic"] = next(
                topic for topic in topic_values if topic_labels[topic] == selected_topic
            )

    if "subject" in missing:
        subject_values = [value for value, _label in SUBJECT_OPTIONS]
        subject_labels = {value: label for value, label in SUBJECT_OPTIONS}
        selected_subject = st.selectbox(
            "Subject",
            ["— Select subject —"] + [subject_labels[value] for value in subject_values],
            key=f"subject_{message_key}",
        )
        if not selected_subject.startswith("—"):
            selections["subject"] = next(
                value for value, label in SUBJECT_OPTIONS if label == selected_subject
            )

    if "question_type" in missing:
        type_values = [value for value, _label in QUESTION_TYPE_OPTIONS]
        type_labels = {value: label for value, label in QUESTION_TYPE_OPTIONS}
        selected_type = st.selectbox(
            "Question type",
            ["— Select question type —"] + [type_labels[value] for value in type_values],
            key=f"type_{message_key}",
        )
        if not selected_type.startswith("—"):
            selections["question_type"] = next(
                value for value, label in QUESTION_TYPE_OPTIONS if label == selected_type
            )

    if "count" in missing:
        count_values = [value for value, _label in COUNT_OPTIONS]
        count_labels = {value: label for value, label in COUNT_OPTIONS}
        count_index = 0
        if partial_intent.get("fetch_all"):
            count_index = count_values.index("all") + 1
        elif partial_intent.get("has_explicit_count") and partial_intent.get("limit") is not None:
            limit_key = str(partial_intent["limit"])
            if limit_key in count_values:
                count_index = count_values.index(limit_key) + 1

        selected_count = st.selectbox(
            "Number of questions",
            ["— Select count —"] + [count_labels[value] for value in count_values],
            index=count_index,
            key=f"count_{message_key}",
        )
        if not selected_count.startswith("—"):
            selections["count_choice"] = next(
                value for value, label in COUNT_OPTIONS if label == selected_count
            )

    if "difficulty" in missing:
        difficulty_values = [value for value, _label in DIFFICULTY_OPTIONS]
        difficulty_labels = {value: label for value, label in DIFFICULTY_OPTIONS}
        difficulty_index = 0
        if partial_intent.get("difficulty") in difficulty_values:
            difficulty_index = difficulty_values.index(partial_intent["difficulty"]) + 1

        selected_difficulty = st.selectbox(
            "Difficulty level",
            ["— Select difficulty —"] + [difficulty_labels[value] for value in difficulty_values],
            index=difficulty_index,
            key=f"difficulty_{message_key}",
        )
        if not selected_difficulty.startswith("—"):
            selections["difficulty"] = next(
                value for value, label in DIFFICULTY_OPTIONS if label == selected_difficulty
            )

    if st.button("Search Questions", type="primary", key=f"search_{message_key}"):
        still_missing = []
        if "topic" in missing and not selections.get("topic"):
            still_missing.append("topic / collection")
        if "subject" in missing and not selections.get("subject"):
            still_missing.append("subject")
        if "question_type" in missing and not selections.get("question_type"):
            still_missing.append("question type")
        if "count" in missing and not selections.get("count_choice"):
            still_missing.append("number of questions")
        if "difficulty" in missing and not selections.get("difficulty"):
            still_missing.append("difficulty level")

        if still_missing:
            st.warning(f"Please choose: {', '.join(still_missing)}.")
            return

        st.session_state.execute_search = {
            "intent": build_intent_from_selection(partial_intent, selections),
            "query": original_query,
        }
        st.rerun()


def run_search_and_store(client, embeddings, intent: dict, original_query: str) -> None:
    query = build_query_from_intent(intent)
    results, search_label = search_all_collections(
        client,
        embeddings,
        query,
        intent_override=intent,
    )

    if not results:
        collections = filter_collections(
            get_searchable_collections(QDRANT_URL, QDRANT_API_KEY),
            intent,
        )
        response = generate_search_intro_llm(
            original_query,
            intent,
            0,
            collections,
            no_results=True,
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        return

    pool = fetch_pool_for_intent(client, embeddings, intent, query) if intent_has_filters(intent) else results
    collections_used = sorted({item["collection"] for item in results})
    search_label = generate_search_intro_llm(
        original_query,
        intent,
        len(results),
        collections_used,
    )
    append_assistant_results(results, search_label, intent, pool, query)


try:
    client = load_qdrant_client()
    embeddings = load_embeddings()
except Exception as exc:
    st.error(f"Failed to connect to the database: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_context" not in st.session_state:
    st.session_state.search_context = None

if st.session_state.get("execute_search"):
    payload = st.session_state.pop("execute_search")
    with st.spinner("Searching Topin database..."):
        try:
            run_search_and_store(client, embeddings, payload["intent"], payload["query"])
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Search failed: {exc}"}
            )
    st.rerun()

for msg_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("results") is not None:
            render_results(
                message["results"],
                message.get("search_label", ""),
                f"history_{msg_index}",
                matched_tags=message.get("matched_tags"),
            )
        elif message.get("selection_prompt"):
            render_selection_prompt(
                message.get("content", ""),
                message.get("partial_intent", {}),
                message.get("original_query", ""),
                f"history_{msg_index}",
            )
        else:
            st.markdown(message.get("content", ""))

if user_query := st.chat_input(
    "Ask naturally — e.g. 'all python coding SET_1 questions' or 'give me git mcqs'"
):
    restore_search_context_from_messages()
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    query_intent = parse_query_intent(user_query)
    context = st.session_state.search_context
    follow_up = handle_follow_up(user_query, context) if context else None

    with st.chat_message("assistant"):
        if follow_up is not None:
            results, search_label = follow_up
            message_key = f"live_{len(st.session_state.messages)}"
            response = render_results(
                results,
                search_label,
                message_key,
                matched_tags=context.get("intent", {}).get("tags"),
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "search_label": search_label,
                    "results": results,
                    "matched_tags": context.get("intent", {}).get("tags"),
                }
            )
        else:
            needs_selection, prompt_message = requires_subject_type_selection(user_query, query_intent)
            if context and is_context_refinement_query(user_query, context):
                needs_selection = False

            if needs_selection:
                render_selection_prompt(
                    prompt_message,
                    query_intent,
                    user_query,
                    f"live_{len(st.session_state.messages)}",
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": prompt_message,
                        "selection_prompt": True,
                        "partial_intent": query_intent,
                        "original_query": user_query,
                    }
                )
            elif context and is_context_refinement_query(user_query, context):
                response = (
                    "I could not apply that filter to your current question list. "
                    "Try: `only SUB_TOPIC_GIT_BASICS subtopic questions`"
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("Searching Topin database..."):
                    try:
                        results, search_label = search_all_collections(
                            client, embeddings, user_query, intent_override=query_intent
                        )
                        if not results:
                            collections = filter_collections(
                                get_searchable_collections(QDRANT_URL, QDRANT_API_KEY),
                                query_intent,
                            )
                            response = generate_search_intro_llm(
                                user_query,
                                query_intent,
                                0,
                                collections,
                                no_results=True,
                            )
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.session_state.search_context = None
                        else:
                            query = build_query_from_intent(query_intent)
                            pool = (
                                fetch_pool_for_intent(client, embeddings, query_intent, query)
                                if intent_has_filters(query_intent)
                                else results
                            )
                            collections_used = sorted({item["collection"] for item in results})
                            search_label = generate_search_intro_llm(
                                user_query,
                                query_intent,
                                len(results),
                                collections_used,
                            )
                            message_key = f"live_{len(st.session_state.messages)}"
                            response = render_results(
                                results,
                                search_label,
                                message_key,
                                matched_tags=query_intent.get("tags"),
                            )
                            update_search_context(query_intent, results, pool, query)
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": response,
                                    "search_label": search_label,
                                    "results": results,
                                    "matched_tags": query_intent.get("tags"),
                                }
                            )
                    except Exception as exc:
                        response = f"Search failed: {exc}"
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

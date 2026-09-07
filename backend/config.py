"""Load secrets from env vars or .streamlit/secrets.toml for non-Streamlit runtimes."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"


def load_secrets() -> dict:
    values: dict = {}
    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("rb") as handle:
            values.update(tomllib.load(handle))

    # Environment overrides (for hosting)
    for key in (
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_CLOUD",
        "PINECONE_REGION",
        "OPENROUTER_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "data_link",
    ):
        if os.getenv(key):
            values[key] = os.environ[key]
    return values


def apply_secrets_to_environ() -> dict:
    secrets = load_secrets()
    for key, value in secrets.items():
        if value is not None and key not in os.environ:
            os.environ[key] = str(value)
    return secrets

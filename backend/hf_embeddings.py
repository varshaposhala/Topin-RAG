"""Embedding backends for query vectors (compatible with all-MiniLM-L6-v2 / 384-dim).

Default for hosting: fastembed (ONNX, low RAM — works on Render free/starter).
Optional: local sentence-transformers, or remote HF Inference API.
"""

from __future__ import annotations

import os
from typing import Protocol


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbeddings:
    """ONNX MiniLM via fastembed — same model family, much less RAM than torch."""

    def __init__(self, model_name: str = MODEL_NAME):
        from fastembed import TextEmbedding

        # Prefer a writable project cache when present (helps Windows/dev).
        cache = os.getenv("FASTEMBED_CACHE_PATH")
        kwargs = {"model_name": model_name}
        if cache:
            kwargs["cache_dir"] = cache
        self.model = TextEmbedding(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        vector = next(self.model.embed([text]))
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(x) for x in row] for row in self.model.embed(texts)]


class LocalSTEmbeddings:
    """Run MiniLM locally with sentence-transformers (heavier; needs ~1GB+ RAM)."""

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [[float(x) for x in row] for row in vectors]


class RemoteHFEmbeddings:
    """Hugging Face Inference API (needs a token with Inference permission)."""

    def __init__(self, model_name: str = MODEL_NAME, token: str | None = None):
        import httpx

        self.model_name = model_name
        self._httpx = httpx
        resolved = (
            token
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if not resolved:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN / HF_TOKEN is missing for remote embeddings."
            )
        self.token = resolved
        self.url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def _post(self, inputs):
        response = self._httpx.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json={"inputs": inputs, "options": {"wait_for_model": True}},
            timeout=60.0,
        )
        if response.status_code == 403:
            raise RuntimeError(
                "Hugging Face token lacks Inference API permission. "
                "Create a token with Inference access at https://huggingface.co/settings/tokens "
                "or set EMBEDDINGS_BACKEND=fast."
            )
        response.raise_for_status()
        return response.json()

    def _as_vector(self, data) -> list[float]:
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return [float(x) for x in data]
        if isinstance(data, list) and data and isinstance(data[0], list):
            dim = len(data[0])
            sums = [0.0] * dim
            for row in data:
                for i, value in enumerate(row):
                    sums[i] += float(value)
            n = max(len(data), 1)
            return [value / n for value in sums]
        raise RuntimeError(f"Unexpected embedding response shape: {type(data)}")

    def embed_query(self, text: str) -> list[float]:
        return self._as_vector(self._post(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def create_embeddings() -> Embeddings:
    """
    EMBEDDINGS_BACKEND:
      - fast (default on Render/Docker): fastembed ONNX, low RAM
      - local: sentence-transformers + torch (~1GB+ RAM)
      - remote: Hugging Face Inference API
      - auto: fast → local → remote
    """
    backend = (os.getenv("EMBEDDINGS_BACKEND") or "fast").strip().lower()

    if backend == "remote":
        return RemoteHFEmbeddings()

    if backend == "local":
        return LocalSTEmbeddings()

    if backend == "auto":
        for factory, name in (
            (FastEmbedEmbeddings, "fast"),
            (LocalSTEmbeddings, "local"),
            (RemoteHFEmbeddings, "remote"),
        ):
            try:
                print(f"[embeddings] trying {name}", flush=True)
                return factory()
            except Exception as exc:  # noqa: BLE001
                print(f"[embeddings] {name} failed: {exc}", flush=True)
        raise RuntimeError("No embedding backend could be loaded.")

    # default: fast (Render-friendly)
    return FastEmbedEmbeddings()

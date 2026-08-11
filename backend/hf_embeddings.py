"""Remote embeddings client using the Hugging Face Inference API.

Avoids importing torch/sentence-transformers in-process, which is too
memory-heavy for small hosting instances (e.g. Render's 512MB tier).
Uses the same model the Pinecone index was built with, so vectors stay
compatible with what's already stored.
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RemoteHFEmbeddings:
    def __init__(self, model_name: str = MODEL_NAME, token: str | None = None):
        self.model_name = model_name
        resolved_token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not resolved_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is missing. Add it to environment variables "
                "(get a free token at https://huggingface.co/settings/tokens)."
            )
        self.client = InferenceClient(model=model_name, token=resolved_token)

    def embed_query(self, text: str) -> list[float]:
        vector = self.client.feature_extraction(text, model=self.model_name)
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

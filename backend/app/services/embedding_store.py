"""Vector store management using FAISS and sentence-transformers."""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from ..models import Document


class EmbeddingStore:
    """Wrap FAISS index creation and similarity search."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:  # pragma: no cover - model download issues
                self.model = None

        self.embedding_dim = (
            self.model.get_sentence_embedding_dimension()
            if self.model is not None
            else 384
        )

        self.index: faiss.Index | None = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided to build the vector store")

        if self.model is not None:
            embeddings = self.model.encode([doc.content for doc in documents], show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
        else:
            embeddings = np.vstack([self._fallback_embed(doc.content) for doc in documents])

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = documents

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.index:
            raise RuntimeError("Vector store not built")

        if self.model is not None:
            query_embedding = self.model.encode([query]).astype(np.float32)
        else:
            query_embedding = np.stack([self._fallback_embed(query)], axis=0)
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))

        results: List[Tuple[Document, float]] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((self.documents[idx], float(distance)))
        return results

    def _fallback_embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.embedding_dim).astype(np.float32)

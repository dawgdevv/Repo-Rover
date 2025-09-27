"""Vector store management using FAISS and sentence-transformers."""

from __future__ import annotations

import hashlib
import logging
from threading import Lock
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from ..models import Document


logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Wrap FAISS index creation and similarity search."""

    _model_cache: Dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]
    _embedding_dim_cache: Dict[str, int] = {}
    _cache_lock: Lock = Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]

        if SentenceTransformer is not None:
            cached = self._get_cached_model(model_name)
            if cached is not None:
                self.model = cached

        self.embedding_dim = (
            self._embedding_dim_cache.get(model_name)
            if self.model is not None
            else 384
        ) or 384

        self.index: faiss.Index | None = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided to build the vector store")

        self._ensure_model_loaded()

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

    def _ensure_model_loaded(self) -> None:
        if self.model is not None or SentenceTransformer is None:
            return

        try:
            model = self._load_or_cache_model(self.model_name)
        except Exception as exc:  # pragma: no cover - model download issues
            logger.warning("Failed to load embedding model '%s': %s", self.model_name, exc)
            self.model = None
            self.embedding_dim = 384
            return

        self.model = model
        self.embedding_dim = self._embedding_dim_cache.get(self.model_name, 384)

    @classmethod
    def _get_cached_model(cls, model_name: str) -> Optional["SentenceTransformer"]:  # type: ignore[name-defined]
        with cls._cache_lock:
            return cls._model_cache.get(model_name)

    @classmethod
    def _load_or_cache_model(cls, model_name: str) -> "SentenceTransformer":  # type: ignore[name-defined]
        cached = cls._get_cached_model(model_name)
        if cached is not None:
            return cached

        model = SentenceTransformer(model_name)  # type: ignore[call-arg]
        dimension = model.get_sentence_embedding_dimension()

        with cls._cache_lock:
            cls._model_cache[model_name] = model
            cls._embedding_dim_cache[model_name] = dimension

        return model

"""Dependency injection helpers for the FastAPI routes."""

from functools import lru_cache

from ..core.config import get_settings
from ..services.rag_service import RAGPipeline


@lru_cache
def get_rag_service() -> RAGPipeline:
    """Return a cached instance of the RAG pipeline service."""

    settings = get_settings()
    return RAGPipeline(settings=settings)

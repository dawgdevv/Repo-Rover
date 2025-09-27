"""Tests for the RAG pipeline with lightweight stubs to avoid heavy downloads."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from backend.app.core.config import Settings
from backend.app.models import Document
from backend.app.schemas.requests import RepositoryAnalysisRequest
from backend.app.services import RAGPipeline


class StubEmbeddingStore:
    """Minimal FAISS replacement for tests."""

    def __init__(self, model_name: str) -> None:  # noqa: D401
        self.model_name = model_name
        self.documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        self.documents = documents

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:  # noqa: D401
        return [(doc, 0.0) for doc in self.documents[:k]]


@pytest.mark.asyncio
async def test_pipeline_generates_artifacts(monkeypatch, tmp_path: Path) -> None:
    repo_dir = tmp_path / "sample"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Sample Repo\n\nSome introductory text", encoding="utf-8")
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "module.py").write_text("print('hello world')", encoding="utf-8")

    settings = Settings(workspace_dir=tmp_path)
    pipeline = RAGPipeline(settings)

    monkeypatch.setattr(pipeline.fetcher, "fetch", lambda *args, **kwargs: repo_dir)
    monkeypatch.setattr("backend.app.services.rag_service.EmbeddingStore", StubEmbeddingStore)

    payload = RepositoryAnalysisRequest(repo_url="https://example.com/org/repo.git")

    result = await pipeline.analyze_repository(payload)

    assert result.repo_url == str(payload.repo_url)
    assert any(artifact.name == "Repository Summary" for artifact in result.artifacts)
    assert "graph TD" in result.mermaid_diagram
    assert result.architecture_map
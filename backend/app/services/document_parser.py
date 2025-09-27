"""Parse repository files into structured documents for embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..models import Document
from ..utils import is_probably_text, matches_any


class DocumentParser:
    """Convert repository files into in-memory document objects."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def parse(
        self,
        include_globs: Iterable[str],
        exclude_globs: Iterable[str],
        max_files: int,
    ) -> List[Document]:
        documents: List[Document] = []

        for file_path in sorted(self.base_path.rglob("*")):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(self.base_path)
            rel_posix = relative_path.as_posix()

            if include_globs and not matches_any(Path(rel_posix), include_globs):
                continue

            if exclude_globs and matches_any(Path(rel_posix), exclude_globs):
                continue

            if not is_probably_text(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            documents.append(
                Document(
                    path=relative_path,
                    content=content,
                    metadata={"source": rel_posix},
                )
            )

            if len(documents) >= max_files:
                break

        return documents

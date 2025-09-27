"""Lightweight document representation used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class Document:
    """A single document extracted from a repository."""

    path: Path
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.path)

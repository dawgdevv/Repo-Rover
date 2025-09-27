"""File filtering utilities to determine which repo contents to ingest."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable


TEXTUAL_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".csv",
    ".tsv",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".cxx",
    ".scala",
}


def matches_any(path: Path, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path.as_posix(), pattern) for pattern in patterns)


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXTUAL_EXTENSIONS:
        return True

    try:
        with path.open("rb") as fh:
            chunk = fh.read(1024)
    except OSError:
        return False

    if b"\x00" in chunk:
        return False

    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

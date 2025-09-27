"""Utilities for cloning or fetching repositories for analysis."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from git import Repo

from ..core.config import Settings


class RepositoryFetcher:
    """Fetch and manage repository worktrees for downstream processing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_repo_path(self, repo_url: str) -> Path:
        repo_hash = hashlib.sha1(repo_url.encode("utf-8"), usedforsecurity=False).hexdigest()
        return self.settings.workspace_dir / repo_hash

    def fetch(self, repo_url: str, branch: Optional[str] = None, refresh: bool = False) -> Path:
        destination = self._resolve_repo_path(repo_url)

        if destination.exists() and refresh:
            shutil.rmtree(destination)

        if destination.exists() and not refresh:
            repo = Repo(destination)
            repo.remote().pull()
        else:
            repo = Repo.clone_from(repo_url, destination)

        if branch:
            repo.git.checkout(branch)

        return destination

"""Request payload definitions for the analysis API."""

from typing import List, Optional

from pydantic import BaseModel, HttpUrl, field_validator


class RepositoryAnalysisRequest(BaseModel):
    """Request body for triggering a repository analysis."""

    repo_url: HttpUrl
    branch: Optional[str] = None
    use_github_api: bool = False
    include_globs: List[str] = ["**/*.md", "**/*.py", "**/*.json", "**/*.yaml", "**/*.yml"]
    exclude_globs: List[str] = ["**/.git/**", "**/node_modules/**", "**/dist/**"]
    refresh: bool = False

    @field_validator("include_globs", "exclude_globs")
    @classmethod
    def ensure_glob_patterns(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Pattern list cannot be empty")
        return value

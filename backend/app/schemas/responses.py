"""Response payload definitions for the analysis API."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Generic representation of an artifact returned by the analysis."""

    name: str
    content: str
    format: str = "markdown"


class ArchitectureNode(BaseModel):
    """Node element describing a component in the repository architecture graph."""

    id: str
    label: str
    description: str
    children: List["ArchitectureNode"] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


ArchitectureNode.model_rebuild()


class RepositoryAnalysisResponse(BaseModel):
    """Top-level response structure for the RAG analysis output."""

    repo_url: str
    artifacts: List[Artifact]
    architecture_map: Dict[str, Any]
    mermaid_diagram: str
    onboarding_guide: str
    change_impact_analysis: str

"""High-level orchestration of the repo ingestion and RAG artifact generation."""

from __future__ import annotations

import json
import textwrap
from collections import Counter, defaultdict
from typing import Dict, List

from anyio import to_thread

from ..core.config import Settings
from ..models import Document
from ..schemas.requests import RepositoryAnalysisRequest
from ..schemas.responses import Artifact, RepositoryAnalysisResponse
from .document_parser import DocumentParser
from .embedding_store import EmbeddingStore
from .llm_client import LLMClient
from .repo_loader import RepositoryFetcher


class RAGPipeline:
    """Coordinates ingestion, vectorization, and artifact generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fetcher = RepositoryFetcher(settings)
        self.llm = LLMClient(settings)

    async def analyze_repository(self, payload: RepositoryAnalysisRequest) -> RepositoryAnalysisResponse:
        repo_path = await to_thread.run_sync(
            self.fetcher.fetch,
            payload.repo_url,
            payload.branch,
            payload.refresh,
        )

        parser = DocumentParser(repo_path)
        documents = await to_thread.run_sync(
            parser.parse,
            payload.include_globs,
            payload.exclude_globs,
            self.settings.max_files,
        )

        if documents:
            vector_store = EmbeddingStore(self.settings.embedding_model)
            await to_thread.run_sync(vector_store.build, documents)
        else:
            vector_store = None

        architecture_map = self._build_architecture_map(documents)
        artifacts = self._generate_artifacts(documents, vector_store, architecture_map)
        mermaid_diagram = self._build_mermaid_diagram(architecture_map)
        onboarding_guide = self._build_onboarding_guide(documents)
        change_impact = self._build_change_impact(documents)

        return RepositoryAnalysisResponse(
            repo_url=str(payload.repo_url),
            artifacts=artifacts,
            architecture_map=architecture_map,
            mermaid_diagram=mermaid_diagram,
            onboarding_guide=onboarding_guide,
            change_impact_analysis=change_impact,
        )

    def _generate_artifacts(
        self,
        documents: List[Document],
        vector_store: EmbeddingStore | None,
        architecture_map: Dict[str, Dict],
    ) -> List[Artifact]:
        summary = self._build_summary(documents)
        onboarding = self._build_onboarding_guide(documents)
        change_impact = self._build_change_impact(documents)

        mermaid = self._build_mermaid_diagram(architecture_map)

        artifacts = [
            Artifact(name="Repository Summary", content=summary, format="markdown"),
            Artifact(name="Architecture Map", content=json.dumps(architecture_map, indent=2), format="json"),
            Artifact(name="Mermaid Diagram", content=mermaid, format="mermaid"),
            Artifact(name="Onboarding Guide", content=onboarding, format="markdown"),
            Artifact(name="Change Impact Analysis", content=change_impact, format="markdown"),
        ]

        if vector_store is not None:
            retrieval_notes = textwrap.dedent(
                f"""
                Vector store constructed with {len(vector_store.documents)} documents using
                model `{vector_store.model_name}`.
                """
            ).strip()
            artifacts.append(
                Artifact(name="Vector Store", content=retrieval_notes, format="markdown")
            )

        return artifacts

    def _build_summary(self, documents: List[Document]) -> str:
        if not documents:
            return "No textual documents were discovered in the repository."

        if self.llm.is_configured:
            top_files = "\n".join(f"- `{doc.path}`" for doc in documents[:10])
            system_prompt = "You summarize codebases for onboarding engineers."
            user_prompt = textwrap.dedent(
                f"""
                Provide a high-level summary of this repository.
                There are {len(documents)} textual documents. Here are sample file paths:\n{top_files}
                """
            )
            return self.llm.generate(system_prompt, user_prompt)

        return self._heuristic_summary(documents)

    def _heuristic_summary(self, documents: List[Document]) -> str:
        readme = next((doc for doc in documents if doc.path.name.lower().startswith("readme")), None)

        extension_counts = Counter(doc.path.suffix.lower() or "<root>" for doc in documents)
        dir_counts = Counter(doc.path.parts[0] if len(doc.path.parts) > 1 else "<root>" for doc in documents)

        top_extensions = "\n".join(
            f"- `{ext}`: {count} file{'s' if count != 1 else ''}"
            for ext, count in extension_counts.most_common(8)
        )

        top_dirs = "\n".join(
            f"- `{directory}`: {count} file{'s' if count != 1 else ''}"
            for directory, count in dir_counts.most_common(8)
        )

        summary_lines = [
            "## Repository Overview",
            f"- Total textual documents processed: {len(documents)}",
            "- Primary languages / file types:",
            top_extensions or "  - Not enough information to detect languages.",
            "- Key directories:",
            top_dirs or "  - Files are mostly at the repository root.",
        ]

        if readme:
            excerpt_lines = [line.rstrip() for line in readme.content.splitlines() if line.strip()][:12]
            if excerpt_lines:
                summary_lines.append("\n### README Highlights")
                summary_lines.append("\n".join(excerpt_lines))

        sample_files = "\n".join(f"- `{doc.path}`" for doc in documents[:10])
        summary_lines.append("\n### Sample Files Considered")
        summary_lines.append(sample_files)

        return "\n".join(summary_lines).strip()

    def _build_architecture_map(self, documents: List[Document]) -> Dict[str, Dict]:
        tree: Dict[str, Dict] = {}

        for doc in documents:
            node = tree
            for part in doc.path.parts[:-1]:
                node = node.setdefault(part, {})
            files = node.setdefault("__files__", [])
            files.append(doc.path.name)

        return tree

    def _build_mermaid_diagram(self, architecture_map: Dict[str, Dict]) -> str:
        lines = ["graph TD", "    Repo[Repository]"]

        def walk(node: Dict, prefix: str) -> None:
            for key, value in list(node.items())[:10]:  # limit for readability
                if key == "__files__":
                    for filename in value[:10]:
                        file_node = f"{prefix}_{filename}".replace("-", "_")
                        lines.append(f"    {prefix} --> {file_node}[{filename}]")
                    continue

                child = f"{prefix}_{key}".replace("-", "_")
                lines.append(f"    {prefix} --> {child}")
                walk(value, child)

        root_label = "Repo"
        walk(architecture_map, root_label)
        return "\n".join(lines)

    def _build_onboarding_guide(self, documents: List[Document]) -> str:
        if not documents:
            return "Repository appears empty; nothing to onboard."

        key_files = "\n".join(f"- `{doc.path}`" for doc in documents[:5])
        return textwrap.dedent(
            f"""
            ## Onboarding Guide

            1. Clone the repository and install dependencies.
            2. Review the primary project files:
{key_files}
            3. Run the automated tests to validate the setup.
            4. Explore remaining modules following the architecture map.
            """
        ).strip()

    def _build_change_impact(self, documents: List[Document]) -> str:
        if not documents:
            return "No changes detected; repository contains no textual files."

        extensions = defaultdict(int)
        for doc in documents:
            extensions[doc.path.suffix or "<root>"] += 1

        extension_summary = "\n".join(f"- `{ext}`: {count} files" for ext, count in extensions.items())
        return textwrap.dedent(
            f"""
            ## Change Impact Considerations

            When modifying this repository, pay attention to the following file type distribution:
{extension_summary}

            Use the vector search endpoint to validate whether changes impact related files.
            """
        ).strip()

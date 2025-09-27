"""Streamlit UI for interacting with the Repo RAG Analyst backend."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
import streamlit as st


def get_backend_url() -> str:
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    try:
        secrets_url = st.secrets["backend_url"]  # type: ignore[index]
    except (FileNotFoundError, KeyError):
        secrets_url = None

    return (secrets_url or "http://localhost:8000").rstrip("/")


def get_request_timeout() -> float:
    env_timeout = os.getenv("BACKEND_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            st.warning("Ignoring invalid BACKEND_TIMEOUT environment variable; using default.")

    try:
        secrets_timeout = st.secrets["backend_timeout"]  # type: ignore[index]
    except (FileNotFoundError, KeyError):
        secrets_timeout = None

    if secrets_timeout is not None:
        try:
            return float(secrets_timeout)
        except (TypeError, ValueError):
            st.warning("Ignoring invalid backend_timeout secret; using default.")

    return 180.0


st.set_page_config(page_title="Repo RAG Analyst", layout="wide")


BACKEND_URL = get_backend_url()
REQUEST_TIMEOUT = get_request_timeout()
API_PATH = "/api/analysis/run"

st.title("🔍 Repo RAG Analyst")
st.caption("Analyze GitHub repositories and generate onboarding artifacts with Retrieval-Augmented Generation.")


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Repository Input")
    repo_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/org/repo")
    branch = st.sidebar.text_input("Branch (optional)")
    use_github_api = st.sidebar.toggle("Use GitHub API", value=False)
    refresh = st.sidebar.toggle("Force refresh", value=False)

    st.sidebar.header("File Filters")
    include_globs = st.sidebar.text_area(
        "Include glob patterns",
        value="**/*.md\n**/*.py\n**/*.json\n**/*.yaml\n**/*.yml",
        help="One pattern per line",
    )
    exclude_globs = st.sidebar.text_area(
        "Exclude glob patterns",
        value="**/.git/**\n**/node_modules/**\n**/dist/**",
        help="One pattern per line",
    )

    return {
        "repo_url": repo_url.strip(),
        "branch": branch.strip() or None,
        "use_github_api": use_github_api,
        "refresh": refresh,
        "include_globs": [pattern.strip() for pattern in include_globs.splitlines() if pattern.strip()],
        "exclude_globs": [pattern.strip() for pattern in exclude_globs.splitlines() if pattern.strip()],
    }


def trigger_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}{API_PATH}",
        json=payload,
        timeout=(10, REQUEST_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def render_artifacts(artifacts: List[Dict[str, Any]]):
    st.header("Generated Artifacts")
    for artifact in artifacts:
        with st.expander(f"{artifact['name']} ({artifact['format']})", expanded=artifact.get("format") == "markdown"):
            if artifact.get("format") == "json":
                st.json(json.loads(artifact["content"]))
            elif artifact.get("format") in {"mermaid", "mermaidjs"}:
                st.markdown(f"```mermaid\n{artifact['content']}\n```")
            else:
                st.markdown(artifact["content"])


def render_architecture_section(architecture_map: Dict[str, Any]):
    st.header("Architecture Map")
    st.json(architecture_map)


payload = render_sidebar()

if st.button("Run Analysis", type="primary", use_container_width=True):
    if not payload["repo_url"]:
        st.error("Please provide a valid repository URL before running the analysis.")
    else:
        with st.spinner(
            "Running RAG pipeline... the first run may take a couple of minutes while models download"
        ):
            try:
                result = trigger_analysis(payload)
            except requests.exceptions.Timeout:
                st.error(
                    "The analysis took too long and timed out. Consider narrowing the file filters or increasing BACKEND_TIMEOUT."
                )
            except requests.HTTPError as exc:
                st.error(f"Request failed: {exc.response.text}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
            else:
                render_artifacts(result.get("artifacts", []))
                if architecture_map := result.get("architecture_map"):
                    render_architecture_section(architecture_map)
                if mermaid_diagram := result.get("mermaid_diagram"):
                    st.header("Mermaid Diagram")
                    st.markdown(f"```mermaid\n{mermaid_diagram}\n```")
                if onboarding := result.get("onboarding_guide"):
                    st.header("Onboarding Guide")
                    st.markdown(onboarding)
                if change_impact := result.get("change_impact_analysis"):
                    st.header("Change Impact Analysis")
                    st.markdown(change_impact)
else:
    st.info("Configure the repo details in the sidebar and press *Run Analysis* to begin.")

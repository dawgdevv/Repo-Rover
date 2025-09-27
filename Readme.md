# Repo RAG Analyst

Retrieval-augmented documentation assistant for GitHub repositories. The system ingests a repository, embeds its contents into FAISS, and produces onboarding-friendly artifacts through an LLM-powered pipeline. The backend is implemented with FastAPI and the web frontend is delivered via Streamlit.

## ✅ Features

- Accepts GitHub repository URLs (with optional branch targeting).
- Clones or refreshes repositories locally using GitPython (GitHub API integration ready).
- Parses source files, docs, and configs with flexible include/exclude glob filters.
- Builds a FAISS-based semantic index using sentence-transformer embeddings (deterministic fallback when models are unavailable).
- Generates:
  - Repository overview summary
  - JSON architecture map
  - Mermaid diagram of the folder structure
  - Onboarding guide
  - Change impact analysis notes
- Exposes artifacts via:
  - REST API (`/api/analysis/run`)
  - Streamlit UI with rich artifact viewers

## 🏗️ Project Structure

```
backend/
   app/
      main.py                # FastAPI application factory
      api/                   # API routers and dependencies
      core/config.py         # Environment-driven settings
      services/              # Repo loader, parser, embeddings, RAG pipeline
      schemas/               # Pydantic request/response models
      utils/                 # File filtering helpers
frontend/
   app.py                   # Streamlit UI
tests/                     # Pytest suite for API + RAG pipeline
```

## 🔧 Requirements

- Python 3.10+
- Git (for repository cloning)
- Optional: OpenAI API key for high-quality artifact generation (`OPENAI_API_KEY`)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Key environment variables (can be placed in `.env`):

- `OPENAI_API_KEY`: Enables real LLM generations (otherwise deterministic fallbacks are used).
- `GITHUB_TOKEN`: Personal access token if you prefer fetching via the GitHub API.
- `WORKSPACE_DIR`: Location to cache cloned repositories (defaults to `.cache/workspace`).
- `EMBEDDING_MODEL`: Sentence-transformer model name (defaults to `sentence-transformers/all-MiniLM-L6-v2`).
- `LLM_MODEL`: Chat completion model to use when `OPENAI_API_KEY` is set (defaults to `gpt-4o-mini`).

## 🚀 Running the Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Interactive docs are served at `http://localhost:8000/api/docs`.

## 🖥️ Running the Streamlit Frontend

```bash
export STREAMLIT_SECRETS='{"backend_url": "http://localhost:8000"}'  # optional override
streamlit run frontend/app.py
```

Alternatively, create a `.streamlit/secrets.toml` file if you prefer configuration via secrets:

```toml
[general]
backend_url = "http://localhost:8000"
```

The UI lets you submit repository URLs, tweak include/exclude patterns, and inspect the generated artifacts.

## 🧪 Tests

```bash
pytest
```

Tests cover the FastAPI health endpoint and the RAG pipeline (using lightweight stubs to avoid large model downloads). If dependency installation is skipped, these tests will not execute.

## 🗺️ Next Steps

- Integrate GitHub API path for private repo access and richer metadata.
- Persist generated artifacts in a database for later retrieval.
- Extend the frontend with artifact comparison and download options.

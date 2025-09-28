Build a system that:

1. Accepts a GitHub repo link as input.
2. Clones the repo locally or fetches via GitHub API.
3. Parses files (README, code, configs).
4. Embeds content into a vector DB (FAISS).
5. Uses RAG with an LLM to generate:
   - Markdown summaries
   - JSON architecture map
   - Mermaid diagrams
   - Onboarding guide
   - Change impact analysis
6. Exposes results via:
   - REST API (FastAPI)
   - Web UI (Streamlit)

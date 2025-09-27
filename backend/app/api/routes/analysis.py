"""Routes for triggering repository analysis and retrieving artifacts."""

from fastapi import APIRouter, Depends

from ..dependencies import get_rag_service
from ...schemas.requests import RepositoryAnalysisRequest
from ...schemas.responses import RepositoryAnalysisResponse
from ...services.rag_service import RAGPipeline

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/run", response_model=RepositoryAnalysisResponse)
async def run_repository_analysis(
    payload: RepositoryAnalysisRequest,
    rag_service: RAGPipeline = Depends(get_rag_service),
) -> RepositoryAnalysisResponse:
    """Execute the end-to-end RAG pipeline for the provided repository."""

    return await rag_service.analyze_repository(payload)

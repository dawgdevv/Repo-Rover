"""Entry point for the FastAPI backend application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.analysis import router as analysis_router
from .core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hooks for startup and shutdown."""

    settings = get_settings()
    app.state.settings = settings
    yield


def create_application() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}

    application.include_router(analysis_router, prefix=settings.api_prefix)

    return application


app = create_application()

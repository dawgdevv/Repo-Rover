"""Basic tests ensuring the FastAPI app boots and exposes health endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture(scope="session")
def app():
    """Return a FastAPI application instance for tests."""
    return create_app()


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """Synchronous test client for exercising the API."""
    with TestClient(app) as c:
        yield c


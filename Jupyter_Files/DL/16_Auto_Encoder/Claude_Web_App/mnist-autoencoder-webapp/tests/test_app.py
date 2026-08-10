"""
test_app.py
------------
Application-level tests: page loads, endpoint exists, and the API rejects
bad input cleanly (no file, wrong extension, corrupt content).

Run with: pytest tests/test_app.py -v
"""

import io

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Reconstruct Signal" in response.data or b"SIGNAL" in response.data


def test_health_endpoint_exists(client):
    response = client.get("/api/health")
    assert response.status_code in (200, 503)


def test_predict_endpoint_missing_file(client):
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "select an image" in data["error"].lower()


def test_predict_endpoint_invalid_extension(client):
    fake_file = (io.BytesIO(b"not a real image"), "malware.exe")
    response = client.post(
        "/api/predict",
        data={"file": fake_file},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "unsupported" in data["error"].lower()


def test_predict_endpoint_corrupt_image(client):
    fake_file = (io.BytesIO(b"this-is-not-png-bytes"), "digit.png")
    response = client.post(
        "/api/predict",
        data={"file": fake_file},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False

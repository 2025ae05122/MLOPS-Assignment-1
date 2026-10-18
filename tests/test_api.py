from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path(
    "artifacts/heart_disease_pipeline.joblib"
)


VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


@pytest.fixture(scope="module")
def client():
    """
    Create the API test client.

    The CI workflow runs src/train.py before Pytest, so the model
    artefact must exist before API tests begin.
    """

    if not MODEL_PATH.exists():
        pytest.fail(
            "Model artefact is missing. "
            "Run `python src/train.py` before Pytest."
        )

    from api.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert (
        body["service"]
        == "Heart Disease Prediction API"
    )

    assert body["status"] == "running"
    assert body["documentation"] == "/docs"
    assert body["health"] == "/health"
    assert body["metrics"] == "/metrics"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True

    assert (
        "heart_disease_pipeline.joblib"
        in body["model_path"]
    )


def test_prediction_endpoint(client):
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]

    assert body["prediction_label"] in [
        "No disease",
        "Disease",
    ]

    assert 0.0 <= body["confidence"] <= 1.0


def test_prediction_response_has_required_fields(
    client,
):
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert set(body.keys()) == {
        "prediction",
        "prediction_label",
        "confidence",
    }


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("age", -1),
        ("age", 121),
        ("sex", 2),
        ("cp", 0),
        ("cp", 5),
        ("trestbps", 0),
        ("chol", -1),
        ("fbs", 2),
        ("restecg", 3),
        ("thalach", 0),
        ("exang", 2),
        ("oldpeak", -0.1),
        ("slope", 0),
        ("slope", 4),
        ("ca", -1),
        ("ca", 5),
        ("thal", 2),
        ("thal", 8),
    ],
)
def test_invalid_values_are_rejected(
    client,
    field,
    invalid_value,
):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload[field] = invalid_value

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("age")

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_non_numeric_value_is_rejected(client):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["chol"] = "not-a-number"

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_metrics_endpoint(client):
    # Generate requests before checking Prometheus counters.
    client.get("/health")

    client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    response = client.get("/metrics")

    assert response.status_code == 200

    assert (
        "text/plain"
        in response.headers["content-type"]
    )

    metrics_text = response.text

    assert (
        "heart_api_requests_total"
        in metrics_text
    )

    assert (
        "heart_api_request_duration_seconds"
        in metrics_text
    )

    assert (
        "heart_api_predictions_total"
        in metrics_text
    )


def test_openapi_documentation_is_available(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200

    openapi_document = response.json()

    assert (
        openapi_document["info"]["title"]
        == "Heart Disease Prediction API"
    )

    assert "/predict" in openapi_document["paths"]
    assert "/health" in openapi_document["paths"]
    assert "/metrics" in openapi_document["paths"]
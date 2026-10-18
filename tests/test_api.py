import logging

import numpy as np
from fastapi.testclient import TestClient

from heart_disease_mlops import api

client = TestClient(api.app)
SAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


class StubModel:
    classes_ = np.array([0, 1])

    def predict(self, rows):
        return np.array([1])

    def predict_proba(self, rows):
        return np.array([[0.2, 0.8]])


def test_homepage_links_to_service_endpoints():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for target in ["/docs#/default/predict_predict_post", "/docs", "/health", "/metrics"]:
        assert f'href="{target}"' in response.text


def test_predict_contract(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: StubModel())

    response = client.post("/predict", json=SAMPLE)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "label": "disease_present",
        "confidence": 0.8,
        "probabilities": {"0": 0.2, "1": 0.8},
    }


def test_predict_rejects_unknown_fields():
    response = client.post("/predict", json={**SAMPLE, "unexpected": 1})
    assert response.status_code == 422


def test_metrics_endpoint_is_available():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text


def test_request_logs_include_id_and_duration_without_patient_data(monkeypatch, caplog):
    monkeypatch.setattr(api, "load_model", StubModel)

    with caplog.at_level(logging.INFO, logger="heart_disease_api"):
        response = client.post("/predict", json=SAMPLE)

    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"request_id={request_id}" in messages
    assert "status=200" in messages
    assert "duration_ms=" in messages
    assert "chol" not in messages
    assert "confidence" not in messages


def test_unexpected_failures_are_logged_without_exception_details(monkeypatch, caplog):
    def broken_loader():
        raise RuntimeError("private patient details")

    monkeypatch.setattr(api, "load_model", broken_loader)
    with caplog.at_level(logging.INFO, logger="heart_disease_api"):
        with TestClient(api.app, raise_server_exceptions=False) as error_client:
            response = error_client.post("/predict", json=SAMPLE)

    assert response.status_code == 500
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "error_type=RuntimeError" in messages
    assert "status=500" in messages
    assert "private patient details" not in messages
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(small_bundle, monkeypatch):
    import app.api as api
    api.get_bundle.cache_clear()
    monkeypatch.setattr(api, "get_bundle", lambda: small_bundle)
    return TestClient(api.app)


PAYLOAD = {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
           "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and 0 <= r.json()["failure_probability"] <= 1


def test_validation_error(client):
    assert client.post("/predict", json={**PAYLOAD, "torque_nm": -3}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[PAYLOAD, {**PAYLOAD, "type": "H"}])
    assert r.status_code == 200 and len(r.json()) == 2

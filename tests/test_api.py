from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, inputs):
        assert inputs.shape == (1, 3, 128, 128)
        return [[0.90] + [0.01] * 10]


def test_health_and_missing_model(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert (
            client.post("/predict", files={"file": ("a.jpg", b"x")}).status_code == 503
        )


def test_predict(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as image:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
            )
        assert response.status_code == 200
        assert response.json() == {"category": "Bread", "confidence": 0.9}
        assert (
            client.post(
                "/predict", files={"file": ("x.jpg", b"not an image")}
            ).status_code
            == 400
        )


def test_invalid_probabilities(monkeypatch):
    class BrokenModel:
        def predict(self, inputs):
            return [[float("nan")] * 11]

    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", BrokenModel())
        with open("tests/fixtures/bread.jpg", "rb") as image:
            assert (
                client.post(
                    "/predict", files={"file": ("bread.jpg", image)}
                ).status_code
                == 500
            )

from fastapi.testclient import TestClient
from api.main import app


def test_local_endpoint_and_limits():
    client = TestClient(app)
    response = client.post("/verify", json={
        "documents": [{"source": "report", "text": "Revenue increased."}],
        "claims": [{"source": "report", "claim": "Revenue increased."}],
    })
    assert response.status_code == 200
    assert response.json()["verdicts"][0]["supported"] is True
    assert response.json()["metrics"]["remote_attempts"] == 0
    assert client.post("/verify", json={"documents": [], "claims": []}).status_code == 422
